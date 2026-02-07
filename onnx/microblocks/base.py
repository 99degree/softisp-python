# microblocks/base.py
import onnx
import onnx.helper as oh
from onnx import TensorProto
from microblocks.registry import Registry
import logging

logging.basicConfig(level=logging.DEBUG)


class BuildResult:
    """
    Container for a microblock build result. Holds raw references (_ref_*)
    produced by the microblock author and the finalized view (nodes, inits,
    vis, inputs, outputs) after finalize_function() is called.

    Sanitization for FunctionProto internals is implemented as a method on
    BuildResult: sanitize_for_stage(...). This method mutates self.func to a
    sanitized copy and returns a rename_map mapping original_inner_name -> sanitized_name.

    New: BuildResult stores `inner_io_info` mapping original inner IO names
    -> {"type": elem_type, "shape": dims} so callers (e.g., build_all.py)
    can use authoritative inner tensor metadata when promoting outputs.
    """

    def __init__(self, outputs=None, nodes=None, inits=None, vis=None,
                 inputs=None, owner=None, debug=False):
        self._ref_outputs = dict(outputs) if outputs else {}
        self._ref_nodes = list(nodes) if nodes else []
        self._ref_inits = list(inits) if inits else []
        self._ref_vis = list(vis) if vis else []
        self._ref_inputs = dict(inputs) if inputs else {}
        self._owner = owner
        self.debug = bool(debug)

        # Function encapsulation bookkeeping
        self.func = None        # onnx.FunctionProto when encapsulated
        self.call = None        # call node placed in top-level graph
        # record original inner IO names (the names used inside the microblock)
        self.func_io_names = {"inputs": [], "outputs": []}

        # runtime containers (populated by _regenerate_internal or finalize)
        self.inits = []
        self.nodes = []
        self.inputs = {}
        self.outputs = {}
        self.vis = []

        # New: authoritative inner IO metadata mapping
        # keys are original inner names (as authored), values are dict {"type": elem_type, "shape": dims}
        self.inner_io_info = {}

        self._regenerate_internal()

    def setOwner(self, owner):
        """
        Set the owning microblock instance (used for marker and registry).
        """
        self._owner = owner
        return self

    def _regenerate_internal(self):
        """
        Rebuild the public view from the raw references.
        """
        self.nodes = list(self._ref_nodes)
        self.inits = list(self._ref_inits)
        self.vis = list(self._ref_vis)
        self.inputs = {i["name"]: i for i in self._ref_inputs.values()}
        self.outputs = {o["name"]: o for o in self._ref_outputs.values()}

    def appendInput(self, name: str, shape=['N', 'C', 'H', 'W'], desc=None,
                    type=TensorProto.FLOAT, source_stage=None, source=None):
        """
        Register an input used by this microblock (author-level metadata).
        """
        resolved_source = None
        if source_stage and source:
            mh = Registry().getInstance().getMapping(source_stage, [])
            resolved_source = mh.getParam(source)
        existing = self._ref_inputs.get(name)
        if existing:
            existing.update({"source": resolved_source, "shape": shape, "type": type, "desc": desc})
            self._regenerate_internal()
            return self
        self._ref_inputs[name] = {"name": name, "shape": shape, "desc": desc,
                                  "type": type, "source": resolved_source}
        self._regenerate_internal()
        return self

    def appendOutput(self, name: str, shape=['N', 'C', 'H', 'W'], desc=None,
                     type=TensorProto.FLOAT, stage=None):
        """
        Register an output produced by this microblock. If `stage` is provided,
        the output name is prefixed with `stage.` to enforce stage scoping.
        """
        resolved = name
        if stage and not resolved.startswith(stage + "."):
            resolved = f"{stage}.{resolved}"
        self._ref_outputs[resolved] = {"name": resolved, "shape": shape, "desc": desc, "type": type}
        self._regenerate_internal()
        return self

    def addOutputs(self, outputs: dict):
        """
        Bulk add outputs. Each entry must include 'name','shape','type'.
        """
        required_fields = {"name", "shape", "type"}
        for key, meta in outputs.items():
            missing = required_fields - set(meta.keys())
            if missing:
                raise ValueError(f"Output '{key}' missing required fields: {missing}")
            name = meta["name"]
            shape = meta["shape"]
            otype = meta["type"]
            desc = meta.get("desc", None)
            self._ref_outputs[name] = {"name": name, "shape": shape, "desc": desc, "type": otype}
        self._regenerate_internal()
        return self

    def _to_function_and_call(self, func_name: str):
        """
        Build a FunctionProto and a call node from the raw references.

        - In debug mode: return raw internals (no FunctionProto).
        - In production: create a FunctionProto whose signature uses the original
          inner IO names (as authored). The function body contains constants
          and the original internal nodes. Sanitization of internals is done
          later by sanitize_for_stage().
        """
        if self.debug:
            return {
                "func": None,
                "call": None,
                "nodes": list(self._ref_nodes),
                "inputs": list(self._ref_inputs.values()),
                "outputs": list(self._ref_outputs.values()),
                "vis": list(self._ref_vis),
                "inits": list(self._ref_inits),
            }

        input_names_raw = [inp["name"] for inp in self._ref_inputs.values()]
        output_names_raw = [out["name"] for out in self._ref_outputs.values()]

        # record inner IO names for later sanitization/bridging
        self.func_io_names = {"inputs": list(input_names_raw), "outputs": list(output_names_raw)}

        # Wrap initializers as Constant nodes inside the function body (raw names)
        const_nodes = []
        for t in self._ref_inits:
            const_nodes.append(
                oh.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[t.name],  # keep raw name inside function body
                    name=f"{t.name}_const",
                    value=t
                )
            )

        # Function body nodes: constants + original internal nodes (no top-level bridges)
        func_nodes = const_nodes + list(self._ref_nodes)

        # Build FunctionProto using original inner IO names
        func = onnx.FunctionProto()
        func.name = func_name
        func.domain = "softisp"
        func.input.extend(input_names_raw)
        func.output.extend(output_names_raw)
        func.node.extend(func_nodes)

        # ValueInfo for the function signature (original inner names)
        vis = []
        for inp in self._ref_inputs.values():
            vis.append(oh.make_tensor_value_info(
                inp["name"],
                inp.get("type", TensorProto.FLOAT),
                inp.get("shape", ["N", "C", "H", "W"])
            ))
        for out in self._ref_outputs.values():
            vis.append(oh.make_tensor_value_info(
                out["name"],
                out.get("type", TensorProto.FLOAT),
                out.get("shape", ["N", "C", "H", "W"])
            ))
        func.value_info.extend(vis)

        func.opset_import.extend([oh.make_operatorsetid("", 16)])

        # Call node: uses the original inner IO names (the function signature)
        call_node = oh.make_node(
            func_name,
            inputs=input_names_raw,
            outputs=output_names_raw,
            domain="softisp"
        )

        return {
            "func": func,
            "call": call_node,
            "nodes": [call_node],
            "inputs": None,
            "outputs": None,
            "vis": None,
            "inits": None,
        }

    def _attach_marker(self):
        """
        Attach a family/version marker node as a self-contained Constant.
        """
        owner = self._owner
        marker_name = f"{owner.__class__.__name__}.{owner.name}.{owner.version}"

        tensor = oh.make_tensor(marker_name, TensorProto.FLOAT, [1], [1.0])
        const_node = oh.make_node(
            "Constant",
            inputs=[],
            outputs=[marker_name + "_marker"],
            name=f"{marker_name}_const",
            value=tensor
        )

        self.nodes.append(const_node)
        self.vis.append(
            oh.make_tensor_value_info(marker_name + "_marker", TensorProto.FLOAT, [1])
        )

    def finalize_function(self, stage_name: str):
        """
        Finalize the BuildResult into either an inlined set of nodes (debug)
        or a FunctionProto + call node (production). Registers outputs in the
        global registry and attaches a marker node.
        """
        func_name = f"{self._owner.__class__.__name__}_{getattr(self._owner, 'version', 'v0')}"
        bundle = self._to_function_and_call(func_name)

        logging.debug("finalize_function: building FunctionProto %s", func_name)
        if self.debug:
            # Debug mode: expose raw internals
            self.func = None
            self.call = None
            self.nodes = bundle["nodes"]
            self.inputs = {i["name"]: i for i in bundle["inputs"] or []}
            self.outputs = {o["name"]: o for o in bundle["outputs"] or []}
            self.vis = bundle["vis"] or []
            self.inits = bundle["inits"] or []
        else:
            # Production mode: encapsulate, no raw I/O promoted
            self.func = bundle.get("func", None)
            self.call = bundle.get("call", None)
            self.nodes = bundle.get("nodes", [])
            # FunctionProto inputs/outputs are the original inner names
            self.inputs = {name: {"name": name} for name in (self.func.input if self.func else [])}
            self.outputs = {name: {"name": name} for name in (self.func.output if self.func else [])}
            self.vis = []   # bounded inside FunctionProto
            self.inits = []  # bounded inside FunctionProto

        # Always attach marker
        self._attach_marker()

        # Register stage outputs for mapping
        MicroblockBase.registry.register_outputs(stage_name, self._owner.__class__.__name__, self.outputs)
        return self

    # -----------------------------
    # Sanitization moved into BuildResult
    # -----------------------------
    def sanitize_for_stage(self, stage: str, stage_class: str,
                           node_output_names: set, promoted_output_names: set):
        """
        Sanitize self.func (FunctionProto) for inclusion in the top-level model.

        - If debug or no function, do nothing and return empty rename_map.
        - Otherwise, produce a sanitized copy of self.func where internal node outputs
          that collide with node_output_names or promoted_output_names are renamed to:
              {stage}.param.{stage_class}.{orig_name}
        - Update value_info entries accordingly.
        - Insert identity bridge nodes inside the function body:
              original_inner_input -> sanitized_internal (if input was renamed)
              sanitized_internal -> original_inner_output (if output was renamed)
        - Ensure function signature value_info reflects the actual inner node types/dims when available.
        - Populate self.inner_io_info mapping original inner names -> {"type": elem_type, "shape": dims}
        - Replace self.func with the sanitized FunctionProto.
        - Return rename_map mapping original_inner_name -> sanitized_internal_name.
        """
        if self.debug or self.func is None:
            return {}

        collisions = set(node_output_names) | set(promoted_output_names)
        f_copy = onnx.FunctionProto()
        f_copy.CopyFrom(self.func)  # work on a copy

        rename_map = {}
        prefix = f"{stage}.param.{stage_class}"

        # 1) Rename internal node outputs and build rename_map
        for node in f_copy.node:
            # outputs
            for i, out in enumerate(list(node.output)):
                if out in rename_map:
                    node.output[i] = rename_map[out]
                elif out in collisions:
                    new_out = f"{prefix}.{out}"
                    node.output[i] = new_out
                    rename_map[out] = new_out
            # inputs referencing renamed outputs
            for i, inp in enumerate(list(node.input)):
                if inp in rename_map:
                    node.input[i] = rename_map[inp]

        # 2) Update value_info entries if present (rename those that refer to internal names)
        new_value_info = []
        for vi in list(f_copy.value_info):
            vi_copy = onnx.ValueInfoProto()
            vi_copy.CopyFrom(vi)
            if vi_copy.name in rename_map:
                vi_copy.name = rename_map[vi_copy.name]
            elif vi_copy.name in collisions:
                new_name = f"{prefix}.{vi_copy.name}"
                vi_copy.name = new_name
                rename_map[vi.name] = new_name
            new_value_info.append(vi_copy)
        del f_copy.value_info[:]
        f_copy.value_info.extend(new_value_info)

        # 3) Insert identity bridges inside the function body:
        bridge_in_nodes = []
        bridge_out_nodes = []

        orig_inputs = list(self.func.input)   # original inner names
        orig_outputs = list(self.func.output)

        for orig_in in orig_inputs:
            internal_name = rename_map.get(orig_in, orig_in)
            if internal_name != orig_in:
                # Identity: orig_in -> internal_name (inside function)
                bridge_in_nodes.append(oh.make_node("Identity", [orig_in], [internal_name],
                                                    name=f"{stage}_in_{internal_name}_id"))

        for orig_out in orig_outputs:
            internal_name = rename_map.get(orig_out, orig_out)
            if internal_name != orig_out:
                # Identity: internal_name -> orig_out (inside function)
                bridge_out_nodes.append(oh.make_node("Identity", [internal_name], [orig_out],
                                                     name=f"{stage}_out_{internal_name}_id"))

        # Reorder nodes: keep Constant nodes first, then bridge_in, then other nodes, then bridge_out
        const_nodes = [n for n in f_copy.node if n.op_type == "Constant"]
        other_nodes = [n for n in f_copy.node if n.op_type != "Constant"]
        f_copy.node.clear()
        f_copy.node.extend(const_nodes)
        f_copy.node.extend(bridge_in_nodes)
        f_copy.node.extend(other_nodes)
        f_copy.node.extend(bridge_out_nodes)

        # 4) Ensure function signature value_info reflects inner node types/dims when available.
        # Build a lookup of available internal ValueInfoProto entries:
        # prefer f_copy.value_info (already renamed) and then the BuildResult's _ref_vis.
        internal_vis = {vi.name: vi for vi in list(f_copy.value_info)}
        for vi in self._ref_vis:
            if vi.name not in internal_vis:
                internal_vis[vi.name] = vi

        # Helper: extract (elem_type, dims) from a ValueInfoProto
        def _vi_to_type_shape(vi_proto):
            if vi_proto is None:
                return None
            if not vi_proto.type.HasField("tensor_type"):
                return None
            tt = vi_proto.type.tensor_type
            elem_type = tt.elem_type
            dims = []
            for d in tt.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(d.dim_value)
                elif d.HasField("dim_param"):
                    dims.append(d.dim_param)
                else:
                    dims.append(None)
            return elem_type, dims

        # Build new signature value_info list: for each original function input/output,
        # prefer internal_vis entry for the internal name (post-rename), then fall back
        # to existing metadata from self._ref_inputs/_ref_outputs or existing f_copy.value_info.
        sig_value_info = []

        # Reset inner_io_info mapping
        self.inner_io_info = {}

        # Inputs
        for orig_in in orig_inputs:
            internal_name = rename_map.get(orig_in, orig_in)
            vi_internal = internal_vis.get(internal_name)
            ts = _vi_to_type_shape(vi_internal)
            if ts is not None:
                elem_type, dims = ts
                sig_value_info.append(oh.make_tensor_value_info(orig_in, elem_type, dims))
                # store authoritative inner info keyed by original inner name
                self.inner_io_info[orig_in] = {"type": elem_type, "shape": dims}
            else:
                # fallback: use metadata from _ref_inputs if available
                meta = self._ref_inputs.get(orig_in)
                if meta and "type" in meta and "shape" in meta:
                    sig_value_info.append(oh.make_tensor_value_info(orig_in, meta["type"], meta["shape"]))
                    self.inner_io_info[orig_in] = {"type": meta["type"], "shape": meta["shape"]}
                else:
                    # last resort: keep any existing value_info in f_copy that matches orig_in (if present)
                    found = None
                    for vi in list(f_copy.value_info):
                        if vi.name == orig_in:
                            found = vi
                            break
                    if found is not None:
                        sig_value_info.append(found)
                        ts2 = _vi_to_type_shape(found)
                        if ts2:
                            self.inner_io_info[orig_in] = {"type": ts2[0], "shape": ts2[1]}
                        else:
                            self.inner_io_info[orig_in] = {"type": TensorProto.FLOAT, "shape": ["N", "C", "H", "W"]}
                    else:
                        # unknown: create a generic float tensor with dynamic dims
                        sig_value_info.append(oh.make_tensor_value_info(orig_in, TensorProto.FLOAT, ["N", "C", "H", "W"]))
                        self.inner_io_info[orig_in] = {"type": TensorProto.FLOAT, "shape": ["N", "C", "H", "W"]}

        # Outputs
        for orig_out in orig_outputs:
            internal_name = rename_map.get(orig_out, orig_out)
            vi_internal = internal_vis.get(internal_name)
            ts = _vi_to_type_shape(vi_internal)
            if ts is not None:
                elem_type, dims = ts
                sig_value_info.append(oh.make_tensor_value_info(orig_out, elem_type, dims))
                self.inner_io_info[orig_out] = {"type": elem_type, "shape": dims}
            else:
                # fallback: use metadata from _ref_outputs if available
                meta = self._ref_outputs.get(orig_out)
                if meta and "type" in meta and "shape" in meta:
                    sig_value_info.append(oh.make_tensor_value_info(orig_out, meta["type"], meta["shape"]))
                    self.inner_io_info[orig_out] = {"type": meta["type"], "shape": meta["shape"]}
                else:
                    # last resort: keep any existing value_info in f_copy that matches orig_out (if present)
                    found = None
                    for vi in list(f_copy.value_info):
                        if vi.name == orig_out:
                            found = vi
                            break
                    if found is not None:
                        sig_value_info.append(found)
                        ts2 = _vi_to_type_shape(found)
                        if ts2:
                            self.inner_io_info[orig_out] = {"type": ts2[0], "shape": ts2[1]}
                        else:
                            self.inner_io_info[orig_out] = {"type": TensorProto.FLOAT, "shape": ["N", "C", "H", "W"]}
                    else:
                        sig_value_info.append(oh.make_tensor_value_info(orig_out, TensorProto.FLOAT, ["N", "C", "H", "W"]))
                        self.inner_io_info[orig_out] = {"type": TensorProto.FLOAT, "shape": ["N", "C", "H", "W"]}

        # Replace function.value_info with signature value_info plus any remaining internal value_info
        remaining_internal_vis = [vi for name, vi in internal_vis.items() if name not in set(orig_inputs + orig_outputs)]
        del f_copy.value_info[:]
        f_copy.value_info.extend(sig_value_info)
        f_copy.value_info.extend(remaining_internal_vis)

        # Replace self.func with sanitized copy
        self.func = f_copy

        logging.debug(f"[SANITIZE] Stage {stage}: renamed {len(rename_map)} internal names; updated signature value_info")
        logging.debug(f"[SANITIZE] Stage {stage}: inner_io_info keys = {list(self.inner_io_info.keys())}")
        return rename_map


class MicroblockBase:
    """
    Abstract base class for microblocks. Provides build entry points and registry access.
    """

    name = "unnamed"
    version = "v0"

    # Shared registry instance
    registry: Registry = Registry.getInstance()

    # Abstract build entry points (to be implemented by subclasses)
    def build_applier(self, stage: str, prev_stages=None):
        raise NotImplementedError

    def build_algo(self, stage: str, prev_stages=None):
        raise NotImplementedError

    def build_coordinator(self, stage: str, prev_stages=None):
        raise NotImplementedError

    def build_test_algo(self, stage: str, prev_stages=None):
        raise NotImplementedError

    # Public entry wrappers that finalize BuildResult
    def get_build_applier(self, stage: str, prev_stages=None) -> BuildResult:
        result = self.build_applier(stage, prev_stages).setOwner(self)
        return result.finalize_function(stage)

    def get_build_algo(self, stage: str, prev_stages=None) -> BuildResult:
        result = self.build_algo(stage, prev_stages).setOwner(self)
        return result.finalize_function(stage)

    def get_build_coordinator(self, stage: str, prev_stages=None) -> BuildResult:
        result = self.build_coordinator(stage, prev_stages).setOwner(self)
        return result.finalize_function(stage)

    def get_build_test_algo(self, stage: str, prev_stages=None) -> BuildResult:
        result = self.build_test_algo(stage, prev_stages).setOwner(self)
        return result.finalize_function(stage)

    def getMapping(self, stage: str, prev_stages=None):
        return MicroblockBase.registry.getMapping(stage, prev_stages)
