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


# Helper to find existing ValueInfoProto by name and extract (elem_type, dims)
def _find_value_info_by_name(name: str, vis_list):
    """
    Return (elem_type, dims) from a ValueInfoProto in vis_list matching name,
    or None if not found.
    dims is a list where each entry is either an int, a dim_param string, or None.
    """
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


def save_model(nodes, inits, vis, graph_inputs, result_outputs,
               out_path, canonical_name, all_function_defs):
    # collect all node output names
    node_output_names = {out for n in nodes for out in getattr(n, "output", [])}

    # Build graph outputs by referencing existing node outputs only
    outputs = []
    for o in result_outputs.values():
        name, otype, oshape = o.get("name"), o.get("type"), o.get("shape")
        if name not in node_output_names:
            logging.warning(f"[SSA FIX] {name} not found in node outputs, skipping promotion")
            continue
        if name.endswith(".shape"):
            elem_type, dims = TensorProto.INT64, [None]
        else:
            elem_type = otype if otype is not None else TensorProto.FLOAT
            dims = oshape if oshape is not None else ["N", "C", "H", "W"]
        outputs.append(oh.make_tensor_value_info(name, elem_type, dims))

    # filter value_info to avoid duplicates with node outputs or promoted outputs
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
        opset_imports=[onnx.helper.make_operatorsetid("", 16)],
        ir_version=11,
    )

    # Attach function protos (they are expected to be sanitized already by microblocks)
    if all_function_defs:
        model.functions.extend(all_function_defs)

    # Ensure opset imports are set as desired
    model.opset_import.clear()
    model.opset_import.extend([
        oh.make_operatorsetid("", 16),
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

    for n in nodes:
        for out in getattr(n, "output", []):
            logging.debug(f"Graph node output: {out}")

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
            else:
                raise ValueError(f"Unknown mode: {mode}")
        except NotImplementedError:
            print(f"[INFO] Skipping stage '{stage_name}' — {mode} not implemented")
            continue

        # store the microblock instance with the result for later per-stage sanitization
        stage_results.append((stage_name, spec, result, mb))

        print(f"Top-level call node: {[getattr(n, 'op_type', '') for n in result.nodes]}")
        print(f"Inputs: {[i['name'] for i in result.inputs.values()]}")
        print(f"Outputs: {[o['name'] for o in result.outputs.values()]}")

        bad = [o['name'] for o in result.outputs.values()
               if not o['name'].startswith(stage_name + ".")]
        if bad:
            raise RuntimeError(f"Output naming violation in stage {stage_name}: {bad}")

        # If this stage produced a FunctionProto (production encapsulation), result.nodes contains the call node.
        # Inline internals only when there is no FunctionProto or when debug mode is requested.
        if result.func is None or result.debug:
            nodes.extend(result._ref_nodes)
            inits.extend(result._ref_inits)
            vis.extend(result._ref_vis)
        else:
            # append the call node(s) to the top-level graph (the function call)
            nodes.extend(result.nodes)

        # For debug mode, promote inputs with value_info
        if result.debug:
            for inp in result.inputs.values():
                # prefer existing ValueInfo if present in vis or graph_inputs, otherwise use explicit metadata
                found = _find_value_info_by_name(inp["name"], vis + graph_inputs)
                if found is not None:
                    elem_type, dims = found
                    graph_inputs.append(oh.make_tensor_value_info(inp["name"], elem_type, dims))
                    logging.debug(f"[DEBUG PROMOTE] Using existing ValueInfo for debug input '{inp['name']}' type={elem_type} shape={dims}")
                else:
                    # require explicit metadata in debug mode
                    if "type" not in inp or "shape" not in inp:
                        raise RuntimeError(f"Missing metadata for debug input '{inp['name']}' (type/shape)")
                    graph_inputs.append(oh.make_tensor_value_info(inp["name"], inp["type"], inp["shape"]))
            inits.extend(result._ref_inits)
            vis.extend(result._ref_vis)

    # 1. List function value_info names for all stage results (diagnostic)
    for entry in stage_results:
        _, _, r, _ = entry
        if r.func is not None:
            logging.debug(f"[DIAG] func {r.func.name} value_info: {[vi.name for vi in r.func.value_info]}")

    # 2. Show whether result.inputs already has type/shape
    for _, _, r, _ in stage_results:
        for n, inp in r.inputs.items():
            logging.debug(f"[DIAG] result.inputs[{n}] keys={list(inp.keys()) if isinstance(inp, dict) else type(inp)}")

    # 3. Show vis names present at promotion time
    logging.debug("[DIAG] vis names before promotion: %s", [vi.name for vi in vis])

    # === Promote dangling inputs (reuse existing ValueInfo if available) ===
    produced_outputs = {meta["name"] for _, _, r, _ in stage_results for meta in r.outputs.values()}

    # build a combined search list of existing value_info and graph_inputs
    existing_vis = list(vis) + list(graph_inputs) + [
        vi for _, _, r, _ in stage_results if r.func is not None for vi in r.func.value_info
    ]

    missing_meta = []
    for _, _, r, _ in stage_results:
        for inp in r.inputs.values():
            name = inp["name"]
            if name in produced_outputs:
                continue

            # 1) prefer an existing ValueInfoProto (from vis or graph_inputs)
            found = _find_value_info_by_name(name, existing_vis)
            if found is not None:
                elem_type, dims = found
                graph_inputs.append(oh.make_tensor_value_info(name, elem_type, dims))
                logging.debug(f"[PROMOTE] Promoted input '{name}' using existing ValueInfo type={elem_type} shape={dims}")
                continue

            # 2) fall back to explicit metadata on the stage result (if present)
            if "type" in inp and "shape" in inp:
                graph_inputs.append(oh.make_tensor_value_info(name, inp["type"], inp["shape"]))
                logging.debug(f"[PROMOTE] Promoted input '{name}' using stage metadata type={inp['type']} shape={inp['shape']}")
                continue

            # 3) nothing found — record and fail later
            missing_meta.append(name)

    if missing_meta:
        raise RuntimeError(
            "Cannot promote dangling inputs: missing type/shape metadata for the following inputs:\n  "
            + "\n  ".join(missing_meta)
        )

    # === Promote dangling outputs ===
    consumed_inputs = {inp["name"] for _, _, r, _ in stage_results for inp in r.inputs.values()}
    logging.debug(f"[DEBUG] Consumed inputs: {list(consumed_inputs)}")

    node_output_names = {out for n in nodes for out in getattr(n, "output", [])}

    if debug_outputs:
        print("\n=== Debug mode: promoting all outputs from all stages ===")
        promoted_outputs = {meta["name"]: meta for _, _, r, _ in stage_results for meta in r.outputs.values()}
    else:
        print("\n=== Prod mode: promoting dangling outputs only ===")
        promoted_outputs = {}
        for _, _, r, _ in stage_results:
            for meta in r.outputs.values():
                out_name = meta["name"]
                if out_name not in consumed_inputs:
                    # Only promote if not already a node output (we will reference node outputs directly)
                    if out_name not in node_output_names:
                        promoted_outputs[out_name] = meta
                        logging.debug(f"[PROMOTE] {out_name} promoted to graph output")
                    else:
                        logging.debug(f"[SKIP] {out_name} skipped (already node output)")
                else:
                    logging.debug(f"[SKIP] {out_name} skipped (consumed downstream)")

    print(f"\nPromoted outputs: {list(promoted_outputs.keys())}")

    # === Filter value_info to avoid SSA duplicates ===
    promoted_output_names = set(promoted_outputs.keys())
    vis = [v for v in vis if v.name not in node_output_names and v.name not in promoted_output_names]
    logging.debug(f"[FILTER] value_info after cleanup: {[v.name for v in vis]}")

    # === Per-stage sanitization and function attachment ===
    # Now that node_output_names and promoted_outputs are known, ask each microblock
    # to sanitize/prepare its FunctionProto (or inline its nodes in debug mode).
    node_output_names = {out for n in nodes for out in getattr(n, "output", [])}
    promoted_output_names = set(promoted_outputs.keys())

    for stage_name, spec, result, mb in stage_results:
        prep = mb.sanitize_and_prepare_function(stage_name, spec["class"], result,
                                                node_output_names, promoted_output_names)
        if prep["mode"] == "inline":
            # inline nodes (debug mode)
            nodes.extend(prep["nodes"])
            inits.extend(prep["inits"])
            vis.extend(prep["vis"])
        elif prep["mode"] == "function":
            # attach sanitized FunctionProto (internal nodes renamed and internal bridges inserted)
            all_function_defs.append(prep["func"])
        else:
            # nothing to do for this stage
            pass

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
