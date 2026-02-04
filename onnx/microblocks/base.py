# microblocks/base.py
import onnx
import onnx.helper as oh
from microblocks.registry import Registry
import inspect

# microblocks/base.py
import onnx
import onnx.helper as oh
from microblocks.registry import Registry
import inspect

def _suffix_function(name: str) -> str:
    return name if name.endswith(".function") else name + ".function"

class BuildResult:
    def __init__(self, outputs=None, nodes=None, inits=None, vis=None,
                 inputs=None, owner=None, debug=False):
        self._ref_outputs = dict(outputs) if outputs else {}
        self._ref_nodes   = list(nodes) if nodes else []
        self._ref_inits   = list(inits) if inits else []
        self._ref_vis     = list(vis) if vis else []
        self._ref_inputs  = dict(inputs) if inputs else {}
        self._owner       = owner
        self.debug        = None

        self.func = None
        self.call = None
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
        Validation: each output spec must contain keys: name, shape, type, desc.
        """
        required_fields = {"name", "shape", "type"}

        for key, meta in outputs.items():
            # Validate required fields
            missing = required_fields - set(meta.keys())
            if missing:
                print(required_fields)
                print(meta.keys())
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
        if True:
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

            # Production mode: build FunctionProto with suffixed I/O only
            input_names_raw  = [inp["name"] for inp in self._ref_inputs.values()]
            output_names_raw = [out["name"] for out in self._ref_outputs.values()]

            input_names_fn   = [_suffix_function(n) for n in input_names_raw]
            output_names_fn  = [_suffix_function(n) for n in output_names_raw]

            # Wrap initializers as Constant nodes inside the function body (raw names)
            const_nodes = []
            for t in self._ref_inits:
                const_nodes.append(
                    oh.make_node(
                        "Constant",
                        inputs=[],
                        outputs=[t.name],  # keep raw name
                        name=f"{t.name}_const",
                        value=t #attributes=[oh.make_attribute("value", t)]
                    )
                )

            # Identity bridges: external suffixed ↔ internal raw
            bridge_in  = [
                oh.make_node("Identity", inputs=[fn], outputs=[raw])
                for raw, fn in zip(input_names_raw, input_names_fn)
            ]
            bridge_out = [
                oh.make_node("Identity", inputs=[raw], outputs=[fn])
                for raw, fn in zip(output_names_raw, output_names_fn)
            ]

            # Function body nodes: constants + bridges + original nodes
            func_nodes = const_nodes + bridge_in + list(self._ref_nodes) + bridge_out

            # Build FunctionProto
            func = onnx.FunctionProto()
            func.name   = func_name
            func.domain = "softisp"
            func.input.extend(input_names_fn)
            func.output.extend(output_names_fn)
            func.node.extend(func_nodes)

            # ValueInfo only for suffixed external I/O
            vis = []
            for inp, fn in zip(self._ref_inputs.values(), input_names_fn):
                vis.append(oh.make_tensor_value_info(
                    fn,
                    inp.get("type", onnx.TensorProto.FLOAT),
                    inp.get("shape", ["N","C","H","W"])
                ))
            for out, fn in zip(self._ref_outputs.values(), output_names_fn):
                vis.append(oh.make_tensor_value_info(
                    fn,
                    out.get("type", onnx.TensorProto.FLOAT),
                    out.get("shape", ["N","C","H","W"])
                ))
            func.value_info.extend(vis)

            func.opset_import.extend([oh.make_operatorsetid("", 13)])

            # Call node: external suffixed inputs → raw outputs
            call_node = oh.make_node(
                func_name,
                inputs=input_names_fn,
                outputs=output_names_fn,
                domain="softisp"
            )

            # Bundle return: only function and call node, no external I/O or inits
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

        print(f"[DEBUG] finalize_function: building FunctionProto {func_name}")
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
