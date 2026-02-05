#!/usr/bin/env python3
import os
import json
import logging
import onnx
import onnx.helper as oh
from onnx import TensorProto
from microblocks.registry import Registry

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(message)s")


def load_manifest(path: str):
    with open(path, "r") as f:
        return json.load(f)


def normalize_hw_symbols(model: onnx.ModelProto):
    def fix_dims(value_info):
        if not value_info.type.HasField("tensor_type"):
            return
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                if dim.dim_param == "h":
                    dim.dim_param = "H"
                elif dim.dim_param == "w":
                    dim.dim_param = "W"
                elif dim.dim_param in ("n", "N"):
                    dim.ClearField("dim_param")
                    dim.dim_value = 1

    for inp in model.graph.input:
        fix_dims(inp)
    for out in model.graph.output:
        fix_dims(out)
    for vi in model.graph.value_info:
        fix_dims(vi)
    for func in model.functions:
        for vi in func.value_info:
            fix_dims(vi)
    return model


def log_graph_names(nodes, inits, vis, inputs, outputs):
    seen = {}

    def check_and_log(name, kind):
        if name in seen:
            print(f"[SSA ERROR] Duplicate name '{name}' found in {kind}, already seen in {seen[name]}")
        else:
            seen[name] = kind

    for n in nodes:
        for out in getattr(n, "output", []):
            check_and_log(out, f"node:{n.op_type}")
    for t in inits:
        check_and_log(t.name, "initializer")
    for v in vis:
        check_and_log(v.name, "value_info")
    for i in inputs:
        check_and_log(i.name, "graph_input")
    for o in outputs:
        check_and_log(o.name, "graph_output")


def sanitize_function_protos(function_protos, node_output_names, promoted_output_names):
    collisions = set(node_output_names) | set(promoted_output_names)
    sanitized = []

    for f in function_protos:
        f_copy = onnx.FunctionProto()
        f_copy.CopyFrom(f)

        rename_map = {}

        new_outputs = []
        for out_name in list(f_copy.output):
            if out_name in collisions:
                new_name = f"{f_copy.name}.{out_name}"
                logging.debug(f"[FUNC SANITIZE] Renaming function output '{out_name}' -> '{new_name}' in function {f_copy.name}")
                rename_map[out_name] = new_name
                new_outputs.append(new_name)
            else:
                new_outputs.append(out_name)

        del f_copy.output[:]
        f_copy.output.extend(new_outputs)

        new_value_info = []
        for vi in list(f_copy.value_info):
            vi_copy = onnx.ValueInfoProto()
            vi_copy.CopyFrom(vi)
            if vi_copy.name in collisions:
                new_name = f"{f_copy.name}.{vi_copy.name}"
                logging.debug(f"[FUNC SANITIZE] Renaming function value_info '{vi_copy.name}' -> '{new_name}' in function {f_copy.name}")
                vi_copy.name = new_name
            elif vi_copy.name in rename_map:
                vi_copy.name = rename_map[vi_copy.name]
            new_value_info.append(vi_copy)

        del f_copy.value_info[:]
        f_copy.value_info.extend(new_value_info)

        for node in f_copy.node:
            for i, out in enumerate(list(node.output)):
                if out in collisions:
                    new_out = f"{f_copy.name}.{out}"
                    logging.debug(f"[FUNC SANITIZE] Renaming node output '{out}' -> '{new_out}' inside function {f_copy.name}")
                    node.output[i] = new_out
                elif out in rename_map:
                    node.output[i] = rename_map[out]
            for i, inp in enumerate(list(node.input)):
                if inp in rename_map:
                    node.input[i] = rename_map[inp]

        sanitized.append(f_copy)

    return sanitized


def _find_value_info_by_name(name: str, vis_list):
    for vi in vis_list:
        if vi.name == name:
            if not vi.type.HasField("tensor_type"):
                return None
            tt = vi.type.tensor_type
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
    return None


def _log_graph_node_outputs_with_type_shape(nodes, inits, vis, func_list):
    _TYPE_MAP = {
        TensorProto.FLOAT: "float",
        TensorProto.UINT8: "uint8",
        TensorProto.INT8: "int8",
        TensorProto.UINT16: "uint16",
        TensorProto.INT16: "int16",
        TensorProto.INT32: "int32",
        TensorProto.INT64: "int64",
        TensorProto.DOUBLE: "double",
        TensorProto.UINT32: "uint32",
        TensorProto.UINT64: "uint64",
        TensorProto.COMPLEX64: "complex64",
        TensorProto.COMPLEX128: "complex128",
        TensorProto.BFLOAT16: "bfloat16",
    }

    def _format_dims_from_value_info(vi):
        dims = []
        if not vi.type.HasField("tensor_type"):
            return dims
        for d in vi.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            elif d.HasField("dim_param"):
                dims.append(d.dim_param)
            else:
                dims.append(None)
        return dims

    def _find_type_shape(name):
        for vi in vis:
            if vi.name == name and vi.type.HasField("tensor_type"):
                elem_type = vi.type.tensor_type.elem_type
                type_str = _TYPE_MAP.get(elem_type, f"elem_type_{elem_type}")
                dims = _format_dims_from_value_info(vi)
                return type_str, dims

        for init in inits:
            if init.name == name:
                elem_type = init.data_type
                type_str = _TYPE_MAP.get(elem_type, f"elem_type_{elem_type}")
                dims = list(init.dims)
                return type_str, dims

        for f in func_list:
            for vi in getattr(f, "value_info", []):
                if vi.name == name and vi.type.HasField("tensor_type"):
                    elem_type = vi.type.tensor_type.elem_type
                    type_str = _TYPE_MAP.get(elem_type, f"elem_type_{elem_type}")
                    dims = _format_dims_from_value_info(vi)
                    return type_str, dims

        return None, None

    for n in nodes:
        for out in getattr(n, "output", []):
            tname, shape = _find_type_shape(out)
            if tname is not None:
                logging.debug(f"Graph node output: {out}  type=tensor({tname})  shape={shape}")
            else:
                logging.debug(f"Graph node output: {out}  type=UNKNOWN  shape=UNKNOWN")


def save_model(nodes, inits, vis, graph_inputs, result_outputs,
               out_path, canonical_name, all_function_defs):
    # collect all node output names
    node_output_names = {out for n in nodes for out in getattr(n, "output", [])}

    # Keep a copy of vis BEFORE we filter it for SSA duplicates.
    # This copy contains authoritative ValueInfoProto entries for nodes/functions.
    vis_before = list(vis)

    # Helper: resolve type/shape from authoritative sources (vis_before, inits, function protos)
    def _resolve_output_type_shape_from_model(name, vis_list, inits_list, func_list):
        # 1) top-level value_info (use vis_before)
        for vi in vis_list:
            if vi.name == name and vi.type.HasField("tensor_type"):
                tt = vi.type.tensor_type
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
        # 2) initializers
        for init in inits_list:
            if init.name == name:
                return init.data_type, list(init.dims)
        # 3) function value_info (search original function protos)
        for f in func_list:
            for vi in getattr(f, "value_info", []):
                if vi.name == name and vi.type.HasField("tensor_type"):
                    tt = vi.type.tensor_type
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
        return None, None

    # Build graph outputs by referencing existing node outputs only,
    # resolving types from vis_before and original function protos.
    outputs = []
    func_list = all_function_defs or []
    for o in result_outputs.values():
        name, otype, oshape = o.get("name"), o.get("type"), o.get("shape")
        if name not in node_output_names:
            logging.warning(f"[SSA FIX] {name} not found in node outputs, skipping promotion")
            continue

        # Resolve using authoritative sources (vis_before, inits, func_list)
        elem_type, dims = _resolve_output_type_shape_from_model(name, vis_before, inits, func_list)

        # Fall back to stage metadata or sensible defaults only if not found
        if elem_type is None:
            if name.endswith(".shape"):
                elem_type, dims = TensorProto.INT64, [None]
            elif otype is not None and oshape is not None:
                elem_type, dims = otype, oshape
            else:
                logging.warning(f"[OUTPUT-FALLBACK] No type/shape found for {name}; using FLOAT fallback")
                elem_type, dims = TensorProto.FLOAT, ["N", "C", "H", "W"]

        outputs.append(oh.make_tensor_value_info(name, elem_type, dims))

    # Now filter vis to avoid duplicates with node outputs or promoted outputs
    vis = [v for v in vis if v.name not in node_output_names and v.name not in result_outputs]

    # Create graph
    graph = oh.make_graph(
        nodes=nodes,
        name=canonical_name,
        inputs=graph_inputs,
        outputs=outputs,
        initializer=inits,
        value_info=vis
    )

    model = oh.make_model(
        graph,
        producer_name="softisp_rewrite",
        opset_imports=[onnx.helper.make_operatorsetid("", 13)],
        ir_version=11,
    )

    # Sanitize and attach function protos if present (this runs after outputs are built)
    if all_function_defs:
        promoted_output_names = set(result_outputs.keys())
        logging.debug("[SANITIZE] Checking function protos for name collisions...")
        safe_funcs = sanitize_function_protos(all_function_defs, node_output_names, promoted_output_names)
        model.functions.extend(safe_funcs)

    # Ensure opset imports are set as desired
    model.opset_import.clear()
    model.opset_import.extend([
        oh.make_operatorsetid("", 13),
        oh.make_operatorsetid("softisp", 1)
    ])

    for f in model.functions:
        logging.debug(f"Function attached: {f.domain}:{f.name}")

    logging.debug("=== Graph Outputs BEFORE model ===")
    for o in result_outputs.values():
        logging.debug(f"Output: {o['name']}")

    logging.debug("=== Final Graph Outputs ===")
    for o in outputs:
        logging.debug(f"Output: {o.name}")

    # Use vis_before when logging node outputs with type/shape so we show authoritative types
    _log_graph_node_outputs_with_type_shape(nodes, inits, vis_before, all_function_defs or [])

    log_graph_names(nodes, inits, vis, graph_inputs, outputs)

    # Validate and save
    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model = normalize_hw_symbols(model)
    onnx.save(model, out_path)
    logging.info(f"Saved ONNX model to {out_path}")


def build_all(manifest_file: str, mode: str = "applier", debug_outputs: bool = False):
    manifest = load_manifest(manifest_file)
    stages_spec = manifest["stages"]

    nodes, inits, vis = [], [], []
    graph_inputs, all_function_defs, stage_results = [], [], []

    reg = Registry().getInstance()
    reg.import_all_microblocks()
    reg.clear_all_outputs()
    reg.set_dynamic_map(stages_spec)

    print("=== Building stages ===")
    for stage_name, spec in stages_spec.items():
        print(f"\n--- Stage {stage_name} ({spec['class']} v{spec['version']}) ---")
        mb_cls = reg.dump_registry()[(spec["class"], spec["version"])]
        mb = mb_cls()

        try:
            if mode == "applier":
                result = mb.get_build_applier(stage_name, prev_stages=spec.get("inputs", []))
            elif mode == "algo":
                result = mb.get_build_algo(stage_name, prev_stages=spec.get("inputs", []))
            elif mode == "test_algo":
                result = mb.get_build_test_algo(stage_name, prev_stages=spec.get("inputs", []))
            elif mode == "coordinator":
                result = mb.get_build_test_algo(stage_name, prev_stages=spec.get("inputs", []))
            else:
                raise ValueError(f"Unknown mode: {mode}")
        except NotImplementedError:
            print(f"[INFO] Skipping stage '{stage_name}' — {mode} not implemented")
            continue

        stage_results.append(result)

        print(f"Top-level call node: {[getattr(n, 'op_type', '') for n in result.nodes]}")
        print(f"Inputs: {[i['name'] for i in result.inputs.values()]}")
        print(f"Outputs: {[o['name'] for o in result.outputs.values()]}")

        bad = [o['name'] for o in result.outputs.values()
               if not o['name'].startswith(stage_name + ".")]
        if bad:
            raise RuntimeError(f"Output naming violation in stage {stage_name}: {bad}")

        nodes.extend(result.nodes)
        inits.extend(result.inits)
        vis.extend(result.vis)

        if result.func is not None:
            all_function_defs.append(result.func)

        if result.debug:
            for inp in result.inputs.values():
                found = _find_value_info_by_name(inp["name"], vis + graph_inputs)
                if found is not None:
                    elem_type, dims = found
                    graph_inputs.append(oh.make_tensor_value_info(inp["name"], elem_type, dims))
                    logging.debug(f"[DEBUG PROMOTE] Using existing ValueInfo for debug input '{inp['name']}' type={elem_type} shape={dims}")
                else:
                    if "type" not in inp or "shape" not in inp:
                        raise RuntimeError(f"Missing metadata for debug input '{inp['name']}' (type/shape)")
                    graph_inputs.append(oh.make_tensor_value_info(inp["name"], inp["type"], inp["shape"]))
            inits.extend(result.inits)
            vis.extend(result.vis)

    # 1. List function value_info names for all stage results
    for r in stage_results:
        if r.func is not None:
            logging.debug(f"[DIAG] func {r.func.name} value_info: {[vi.name for vi in r.func.value_info]}")

    # 2. Show whether result.inputs already has type/shape
    for r in stage_results:
        for n, inp in r.inputs.items():
            logging.debug(f"[DIAG] result.inputs[{n}] keys={list(inp.keys()) if isinstance(inp, dict) else type(inp)}")

    # 3. Show vis names present at promotion time
    logging.debug("[DIAG] vis names before promotion: %s", [vi.name for vi in vis])

    # === Promote dangling inputs (reuse existing ValueInfo if available) ===
    produced_outputs = {meta["name"] for r in stage_results for meta in r.outputs.values()}

    # build a combined search list of existing value_info and graph_inputs and function value_info
    existing_vis = list(vis) + list(graph_inputs) + [vi for r in stage_results if r.func is not None for vi in r.func.value_info]

    missing_meta = []
    for r in stage_results:
        for inp in r.inputs.values():
            name = inp["name"]
            if name in produced_outputs:
                continue

            found = _find_value_info_by_name(name, existing_vis)
            if found is not None:
                elem_type, dims = found
                graph_inputs.append(oh.make_tensor_value_info(name, elem_type, dims))
                logging.debug(f"[PROMOTE] Promoted input '{name}' using existing ValueInfo type={elem_type} shape={dims}")
                continue

            if "type" in inp and "shape" in inp:
                graph_inputs.append(oh.make_tensor_value_info(name, inp["type"], inp["shape"]))
                logging.debug(f"[PROMOTE] Promoted input '{name}' using stage metadata type={inp['type']} shape={inp['shape']}")
                continue

            missing_meta.append(name)

    if missing_meta:
        raise RuntimeError(
            "Cannot promote dangling inputs: missing type/shape metadata for the following inputs:\n  "
            + "\n  ".join(missing_meta)
        )

    # === Promote dangling outputs ===
    consumed_inputs = {inp["name"] for r in stage_results for inp in r.inputs.values()}
    logging.debug(f"[DEBUG] Consumed inputs: {list(consumed_inputs)}")

    node_output_names = {out for n in nodes for out in getattr(n, "output", [])}

    if debug_outputs:
        print("\n=== Debug mode: promoting all outputs from all stages ===")
        promoted_outputs = {meta["name"]: meta for r in stage_results for meta in r.outputs.values()}
    else:
        print("\n=== Prod mode: promoting dangling outputs only ===")
        promoted_outputs = {}
        for r in stage_results:
            for meta in r.outputs.values():
                out_name = meta["name"]
                if out_name not in consumed_inputs:
                    if out_name not in node_output_names:
                        promoted_outputs[out_name] = meta
                        logging.debug(f"[PROMOTE] {out_name} promoted to graph output")
                    else:
                        logging.debug(f"[SKIP] {out_name} skipped (already node output)")
                else:
                    logging.debug(f"[SKIP] {out_name} skipped (consumed downstream)")

    if not promoted_outputs:
        logging.warning("[PROMOTE] No promoted outputs found; promoting dangling stage outputs as graph outputs")

        candidates = []
        for r in stage_results:
            for meta in r.outputs.values():
                out_name = meta["name"]
                if out_name not in consumed_inputs:
                    candidates.append((out_name, meta))

        for name, meta in candidates:
            lname = name.lower()
            if "marker" in lname:
                logging.debug(f"[PROMOTE-BRANCH] Skipping marker-like output: {name}")
                continue
            if name.endswith(".function"):
                promoted_outputs[name] = meta
                logging.debug(f"[PROMOTE-BRANCH] Promoted dangling stage output (function): {name}")

        if not promoted_outputs:
            for name, meta in candidates:
                lname = name.lower()
                if "marker" in lname:
                    continue
                promoted_outputs[name] = meta
                logging.debug(f"[PROMOTE-BRANCH] Promoted dangling stage output (fallback): {name}")

        if not promoted_outputs:
            logging.warning("[PROMOTE-BRANCH] No non-marker dangling stage outputs found to promote")

    print(f"\nPromoted outputs: {list(promoted_outputs.keys())}")

    promoted_output_names = set(promoted_outputs.keys())
    vis = [v for v in vis if v.name not in node_output_names and v.name not in promoted_output_names]
    logging.debug(f"[FILTER] value_info after cleanup: {[v.name for v in vis]}")

    out_path = os.path.join("onnx_out", manifest["canonical_name"], f"{mode}.onnx")
    save_model(nodes, inits, vis, graph_inputs,
               promoted_outputs,
               out_path, manifest["canonical_name"], all_function_defs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="path to manifest json")
    parser.add_argument("--mode", default="applier", help="applier|algo|test_algo")
    parser.add_argument("--debug-outputs", action="store_true", help="promote all outputs for debugging")
    args = parser.parse_args()
    build_all(args.manifest, args.mode, args.debug_outputs)
