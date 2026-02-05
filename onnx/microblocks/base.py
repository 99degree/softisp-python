# microblocks/base.py
import onnx
import onnx.helper as oh
from microblocks.registry import Registry
import inspect
import logging

logging.basicConfig(level=logging.DEBUG)


class BuildResult:
    def __init__(self, outputs=None, nodes=None, inits=None, vis=None,
                 inputs=None, owner=None, debug=False):
        self._ref_outputs = dict(outputs) if outputs else {}
        self._ref_nodes   = list(nodes) if nodes else []
        self._ref_inits   = list(inits) if inits else []
        self._ref_vis     = list(vis) if vis else []
        self._ref_inputs  = dict(inputs) if inputs else {}
        self._owner       = owner
        # use the passed debug flag
        self.debug        = bool(debug)

        # Function encapsulation bookkeeping
        self.func = None
        self.call = None
        # record original inner IO names (the names used inside the microblock)
        self.func_io_names = {"inputs": [], "outputs": []}

        # runtime containers (populated by _regenerate_internal or finalize)
        self.inits = []
        self.nodes = []
        self.inputs = {}
        self.outputs = {}
        self.vis = []

        self._regenerate_internal()

    def setOwner(self, owner):
        """
        Explicitly set the owning microblock for this BuildResult.
        Must be called before finalize_function so markers can be attached.
        """
        self._owner = owner
        return self

    def _regenerate_internal(self):
        self.nodes   = list(self._ref_nodes)
        self.inits   = list(self._ref_inits)
        self.vis     = list(self._ref_vis)
        self.inputs  = {i["name"]: i for i in self._ref_inputs.values()}
        self.outputs = {o["name"]: o for o in self._ref_outputs.values()}

    def appendInput(self, name: str, shape=['N','C','H','W'], desc=None,
                    type=onnx.TensorProto.FLOAT, source_stage=None, source=None):
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

    def appendOutput(self, name: str, shape=['N','C','H','W'], desc=None,
                     type=onnx.TensorProto.FLOAT, stage=None):
        resolved = name
        if stage and not resolved.startswith(stage + "."):
            resolved = f"{stage}.{resolved}"
        self._ref_outputs[resolved] = {"name": resolved, "shape": shape, "desc": desc, "type": type}
        self._regenerate_internal()
        return self

    def addOutputs(self, outputs: dict):
        """
        Accept a dict of output specs and register them like appendOutput().
        Validation: each output spec must contain keys: name, shape, type.
        """
        required_fields = {"name", "shape", "type"}

        for key, meta in outputs.items():
            # Validate required fields
            missing = required_fields - set(meta.keys())
            if missing:
                raise ValueError(
                    f"Output '{key}' missing required fields: {missing}"
                )

            name   = meta["name"]
            shape  = meta["shape"]
            otype  = meta["type"]
            desc   = meta.get("desc", None)

            self._ref_outputs[name] = {
                "name": name,
                "shape": shape,
                "desc": desc,
                "type": otype,
            }

        self._regenerate_internal()
        return self

    def _to_function_and_call(self, func_name: str):
        """
        Build either:
         - debug bundle (no encapsulation): return raw nodes/inputs/outputs/inits/vis
         - production FunctionProto bundle:
             * FunctionProto where FunctionProto.input/output are the original inner IO names
             * Function body contains constants + original internal nodes (no internal identity bridges)
             * call_node uses the original inner IO names
           The per-stage sanitizer will later rename internal node IOs and insert identity bridges
           mapping original inner IO names <-> sanitized internal names.
        """
        if self.debug:
            # Debug mode: return raw graph parts immediately, no encapsulation
            return {
                "func": None,
                "call": None,
                "nodes": list(self._ref_nodes),
                "inputs": list(self._ref_inputs.values()),
                "outputs": list(self._ref_outputs.values()),
                "vis": list(self._ref_vis),
                "inits": list(self._ref_inits),
            }

        # Production mode: build FunctionProto using original inner IO names
        input_names_raw  = [inp["name"] for inp in self._ref_inputs.values()]
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

        # Function body nodes: constants + original internal nodes (no internal bridges here)
        func_nodes = const_nodes + list(self._ref_nodes)

        # Build FunctionProto
        func = onnx.FunctionProto()
        func.name   = func_name
        func.domain = "softisp"
        # IMPORTANT: use original inner IO names as FunctionProto inputs/outputs
        func.input.extend(input_names_raw)
        func.output.extend(output_names_raw)
        func.node.extend(func_nodes)

        # ValueInfo only for the function's external I/O (we keep original inner names here)
        vis = []
        for inp in self._ref_inputs.values():
            vis.append(oh.make_tensor_value_info(
                inp["name"],
                inp.get("type", onnx.TensorProto.FLOAT),
                inp.get("shape", ["N","C","H","W"])
            ))
        for out in self._ref_outputs.values():
            vis.append(oh.make_tensor_value_info(
                out["name"],
                out.get("type", onnx.TensorProto.FLOAT),
                out.get("shape", ["N","C","H","W"])
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

        # Bundle return: function proto and a single call node (call node will be placed in top-level graph)
        return {
            "func":    func,
            "call":    call_node,
            "nodes":   [call_node],
            "inputs":  None,
            "outputs": None,
            "vis":     None,
            "inits":   None,
        }

    def _attach_marker(self):
        """
        Attach a family/version marker node as a self-contained Constant.
        """
        owner = self._owner
        marker_name = f"{owner.__class__.__name__}.{owner.name}.{owner.version}"

        # Constant node produces the marker output directly
        tensor = oh.make_tensor(marker_name, onnx.TensorProto.FLOAT, [1], [1.0])
        const_node = oh.make_node(
            "Constant",
            inputs=[],
            outputs=[marker_name + "_marker"],
            name=f"{marker_name}_const",
            value=tensor
        )

        # Register the marker node and its value_info
        self.nodes.append(const_node)
        self.vis.append(
            oh.make_tensor_value_info(marker_name + "_marker", onnx.TensorProto.FLOAT, [1])
        )

    def finalize_function(self, stage_name: str):
        func_name = f"{self._owner.__class__.__name__}_{getattr(self._owner, 'version', 'v0')}"
        bundle = self._to_function_and_call(func_name)

        logging.debug("finalize_function: building FunctionProto %s", func_name)
        if self.debug:
            # Debug mode: expose raw internals
            self.func    = None
            self.call    = None
            self.nodes   = bundle["nodes"]
            self.inputs  = {i["name"]: i for i in bundle["inputs"] or []}
            self.outputs = {o["name"]: o for o in bundle["outputs"] or []}
            self.vis     = bundle["vis"] or []
            self.inits   = bundle["inits"] or []
        else:
            # Production mode: encapsulate, no raw I/O promoted
            self.func    = bundle.get("func", None)
            self.call    = bundle.get("call", None)
            self.nodes   = bundle.get("nodes", [])
            # FunctionProto inputs/outputs are the original inner names
            self.inputs = {name: {"name": name} for name in (self.func.input if self.func else [])}
            self.outputs = {name: {"name": name} for name in (self.func.output if self.func else [])}
            self.vis     = []   # bounded inside FunctionProto
            self.inits   = []   # bounded inside FunctionProto

        # Always attach marker
        self._attach_marker()

        # Register stage for later ref
        MicroblockBase.registry.register_outputs(stage_name, self._owner.__class__.__name__, self.outputs)
        return self


class MicroblockBase:
    """
    Abstract base class for all microblocks.
    Each block builds a BuildResult, which handles its own finalization
    (encapsulation, markers, registry, debug/production split).
    """

    name = "unnamed"
    version = "v0"

    # Shared registry instance
    registry: Registry = Registry.getInstance()

    # -----------------------------
    # Abstract build entry points
    # -----------------------------
    def build_applier(self, stage: str, prev_stages=None):
        raise NotImplementedError

    def build_algo(self, stage: str, prev_stages=None):
        raise NotImplementedError

    def build_coordinator(self, stage: str, prev_stages=None):
        raise NotImplementedError

    def build_test_algo(self, stage: str, prev_stages=None):
        raise NotImplementedError

    # -----------------------------
    # Public entry points
    # -----------------------------
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

    # -----------------------------
    # Sanitizer helpers (per-stage)
    # -----------------------------
    def _sanitize_function_proto_for_stage(self, func: onnx.FunctionProto, stage: str, stage_class: str,
                                           node_output_names: set, promoted_output_names: set):
        """
        Sanitize a FunctionProto for inclusion in the top-level model.

        - Rename internal node outputs/inputs that collide with top-level node outputs
          or promoted outputs to a deterministic stage-scoped name:
            {stage}.param.{stage_class}.{orig_name}
        - Insert identity bridge nodes inside the function body that map:
            original_inner_input_name -> sanitized_internal_name
            sanitized_internal_name -> original_inner_output_name
          so the FunctionProto signature remains the original inner names while the
          internal nodes use sanitized names.
        - Return (sanitized_func, bridge_nodes, rename_map)
        """
        collisions = set(node_output_names) | set(promoted_output_names)
        f_copy = onnx.FunctionProto()
        f_copy.CopyFrom(func)

        rename_map = {}
        prefix = f"{stage}.param.{stage_class}"

        # 1) Rename internal node outputs/inputs and build rename_map
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

        # 2) Update value_info entries if present
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
        #    For each original function input name, create Identity(original_input -> internal_sanitized)
        #    For each original function output name, create Identity(internal_sanitized -> original_output)
        bridge_in_nodes = []
        bridge_out_nodes = []

        # function.input and function.output are the original inner names (as built earlier)
        orig_inputs = list(func.input)
        orig_outputs = list(func.output)

        for orig_in in orig_inputs:
            internal_name = rename_map.get(orig_in, orig_in)
            # Identity: orig_in -> internal_name (only if renamed)
            if internal_name != orig_in:
                bridge_in_nodes.append(oh.make_node("Identity", [orig_in], [internal_name],
                                                    name=f"{stage}_in_{internal_name}_id"))

        for orig_out in orig_outputs:
            internal_name = rename_map.get(orig_out, orig_out)
            # Identity: internal_name -> orig_out (only if renamed)
            if internal_name != orig_out:
                bridge_out_nodes.append(oh.make_node("Identity", [internal_name], [orig_out],
                                                     name=f"{stage}_out_{internal_name}_id"))

        # Prepend bridge_in_nodes and append bridge_out_nodes around existing nodes
        const_nodes = [n for n in f_copy.node if n.op_type == "Constant"]
        other_nodes = [n for n in f_copy.node if n.op_type != "Constant"]
        f_copy.node.clear()
        f_copy.node.extend(const_nodes)
        f_copy.node.extend(bridge_in_nodes)
        f_copy.node.extend(other_nodes)
        f_copy.node.extend(bridge_out_nodes)

        # Build top-level bridge nodes to be added to the outer graph
        # These map the top-level names (original inner names) to the sanitized internals
        top_level_bridges = []
        for orig_in in orig_inputs:
            internal_name = rename_map.get(orig_in, orig_in)
            if internal_name != orig_in:
                # top-level: orig_in -> internal_name
                top_level_bridges.append(oh.make_node("Identity", [orig_in], [internal_name],
                                                      name=f"{stage}_top_in_{internal_name}_id"))
        for orig_out in orig_outputs:
            internal_name = rename_map.get(orig_out, orig_out)
            if internal_name != orig_out:
                # top-level: internal_name -> orig_out
                top_level_bridges.append(oh.make_node("Identity", [internal_name], [orig_out],
                                                      name=f"{stage}_top_out_{internal_name}_id"))

        return f_copy, top_level_bridges, rename_map

    def sanitize_and_prepare_function(self, stage: str, stage_class: str, result: BuildResult,
                                      node_output_names: set, promoted_output_names: set):
        """
        Public helper used by build_all:
        - If result.debug is True -> return inline parts (mode='inline')
        - Else -> return sanitized FunctionProto (mode='function') and top-level bridge nodes
        """
        if result.debug:
            # Inline debug mode: return raw nodes/inits/vis to be appended to top-level graph
            return {
                "mode": "inline",
                "nodes": list(result._ref_nodes),
                "inits": list(result._ref_inits),
                "vis": list(result._ref_vis),
                "rename_map": {}
            }

        if result.func is None:
            return {"mode": "none"}

        # Ensure FunctionProto uses original inner IO names (we recorded them earlier)
        if result.func_io_names and result.func_io_names["inputs"]:
            del result.func.input[:]
            result.func.input.extend(result.func_io_names["inputs"])
        if result.func_io_names and result.func_io_names["outputs"]:
            del result.func.output[:]
            result.func.output.extend(result.func_io_names["outputs"])

        sanitized_func, top_level_bridges, rename_map = self._sanitize_function_proto_for_stage(
            result.func, stage, stage_class, node_output_names, promoted_output_names
        )
        return {
            "mode": "function",
            "func": sanitized_func,
            "top_level_bridges": top_level_bridges,
            "rename_map": rename_map
        }
