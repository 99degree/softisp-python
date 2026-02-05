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

    # Attach function protos (they are expected to be sanitized inside the FunctionProto)
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

        # keep result and microblock for later promotion checks
        stage_results.append((stage_name, spec, result, mb))

        print(f"Top-level call node: {[getattr(n, 'op_type', '') for n in result.nodes]}")
        print(f"Inputs: {[i['name'] for i in result.inputs.values()]}")
        print(f"Outputs: {[o['name'] for o in result.outputs.values()]}")

        bad = [o['name'] for o in result.outputs.values()
               if not o['name'].startswith(stage_name + ".")]
        if bad:
            raise RuntimeError(f"Output naming violation in stage {stage_name}: {bad}")

        # Inline internals for debug or when no FunctionProto was produced.
        # Otherwise collect call node and function proto for later processing.
        if result.func is None or result.debug:
            nodes.extend(result._ref_nodes)
            inits.extend(result._ref_inits)
            vis.extend(result._ref_vis)
        else:
            # collect the FunctionProto for later attachment
            if result.func is not None:
                all_function_defs.append(result.func)
            # collect call node(s) in result.nodes but defer final syncing/append until after optional sanitization
            # (we will append call nodes to `nodes` in a dedicated step below, after optional sanitize_for_stage)
            # keep them out of nodes for now to allow sanitization to run first if present
            pass

        # For debug mode, promote inputs with value_info
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
            inits.extend(result._ref_inits)
            vis.extend(result._ref_vis)

    # Diagnostics
    for _, _, r, _ in stage_results:
        if r.func is not None:
            logging.debug(f"[DIAG] func {r.func.name} value_info: {[vi.name for vi in r.func.value_info]}")

    for _, _, r, _ in stage_results:
        for n, inp in r.inputs.items():
            logging.debug(f"[DIAG] result.inputs[{n}] keys={list(inp.keys()) if isinstance(inp, dict) else type(inp)}")

    logging.debug("[DIAG] vis names before promotion: %s", [vi.name for vi in vis])

    # === Promote dangling inputs (reuse existing ValueInfo if available) ===
    produced_outputs = {meta["name"] for _, _, r, _ in stage_results for meta in r.outputs.values()}

    existing_vis = list(vis) + list(graph_inputs) + [
        vi for _, _, r, _ in stage_results if r.func is not None for vi in r.func.value_info
    ]

    missing_meta = []
    for _, _, r, _ in stage_results:
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

    # === Optional per-stage sanitization (if BuildResult provides it) ===
    # If BuildResult implements sanitize_for_stage(stage, stage_class, node_output_names, promoted_output_names),
    # call it now so the FunctionProto internals are adjusted before we append call nodes.
    # We compute a preliminary node_output_names from currently inlined nodes (debug/inlined ones).
    prelim_node_output_names = {out for n in nodes for out in getattr(n, "output", [])}
    prelim_promoted_output_names = set()  # none yet
    for stage_name, spec, result, mb in stage_results:
        if getattr(result, "func", None) is None or result.debug:
            continue
        sanitize_fn = getattr(result, "sanitize_for_stage", None)
        if callable(sanitize_fn):
            logging.debug(f"[SANITIZE] Calling sanitize_for_stage for {stage_name}")
            try:
                # sanitizer returns rename_map or mutates result.func in-place
                result.sanitize_for_stage(stage_name, spec["class"], prelim_node_output_names, prelim_promoted_output_names)
            except Exception as e:
                logging.warning(f"[SANITIZE] sanitize_for_stage failed for {stage_name}: {e}")

    # === Append call nodes (synchronized to sanitized function signatures) ===
    # Ensure call nodes match function signatures and append them to nodes so their outputs are visible.
    for stage_name, spec, result, mb in stage_results:
        if getattr(result, "func", None) is None or result.debug:
            # already inlined or no function
            continue

        # Ensure call node exists
        if getattr(result, "call", None) is None:
            logging.debug(f"[WARN] No call node for stage {stage_name}; skipping call append")
            continue

        # Sync call node signature to function signature (important if sanitizer changed func)
        try:
            # inputs
            del result.call.input[:]
            result.call.input.extend(result.func.input)
            # outputs
            del result.call.output[:]
            result.call.output.extend(result.func.output)
        except Exception as e:
            logging.warning(f"[SYNC] Failed to sync call node for {stage_name}: {e}")

        # Append call node(s) to top-level nodes (avoid duplicates)
        # Use simple identity check by op_type + outputs to avoid duplicate append
        already = False
        for n in nodes:
            if getattr(n, "op_type", None) == getattr(result.call, "op_type", None) and list(n.output) == list(result.call.output):
                already = True
                break
        if not already:
            nodes.append(result.call)

    # After appending call nodes, collect all FunctionProto defs for attachment (they were collected earlier)
    # all_function_defs already contains result.func for each non-debug function result (collected above)

    # === Promote dangling outputs (stage-declared outputs) ===
    # Now that call nodes are present, compute node_output_names and promote any stage-declared output
    # that is not consumed downstream. This includes outputs produced by call nodes.
    node_output_names = {out for n in nodes for out in getattr(n, "output", [])}
    consumed_inputs = {inp["name"] for _, _, r, _ in stage_results for inp in r.inputs.values()}
    logging.debug(f"[DEBUG] Consumed inputs: {list(consumed_inputs)}")

    if debug_outputs:
        print("\n=== Debug mode: promoting all outputs from all stages ===")
        promoted_outputs = {meta["name"]: meta for _, _, r, _ in stage_results for meta in r.outputs.values()}
    else:
        print("\n=== Prod mode: promoting dangling outputs only ===")
        promoted_outputs = {}
        for _, _, r, _ in stage_results:
            for meta in r.outputs.values():
                out_name = meta["name"]
                if out_name in consumed_inputs:
                    logging.debug(f"[SKIP] {out_name} skipped (consumed downstream)")
                    continue
                # Promote any stage-declared output that is not consumed downstream.
                # This includes outputs that are already produced by top-level nodes (call nodes).
                promoted_outputs[out_name] = meta
                logging.debug(f"[PROMOTE] {out_name} promoted to graph output (declared by stage)")

    logging.debug(f"\nPromoted outputs (pre-function-check): {list(promoted_outputs.keys())}")

    # === Filter value_info to avoid SSA duplicates ===
    promoted_output_names = set(promoted_outputs.keys())
    vis = [v for v in vis if v.name not in node_output_names and v.name not in promoted_output_names]
    logging.debug(f"[FILTER] value_info after cleanup: {[v.name for v in vis]}")

    # === Promote any remaining function outputs that are not covered yet ===
    # (This is a safety net: if a FunctionProto declares outputs that were not in stage outputs,
    #  promote them if they are dangling and not consumed.)
    node_output_names = {out for n in nodes for out in getattr(n, "output", [])}
    consumed_inputs = {inp["name"] for _, _, r, _ in stage_results for inp in r.inputs.values()}

    for func in all_function_defs:
        for out_name in list(func.output):
            if out_name in promoted_outputs:
                continue
            if out_name in node_output_names:
                # already produced by a top-level node (call node) — but if the stage didn't declare it, still promote if dangling
                if out_name in consumed_inputs:
                    continue
                # promote it (use function.value_info if available)
                found = _find_value_info_by_name(out_name, func.value_info)
                if found is not None:
                    elem_type, dims = found
                    promoted_outputs[out_name] = {"name": out_name, "type": elem_type, "shape": dims}
                    logging.debug(f"[PROMOTE-FUNC] Promoting function output '{out_name}' using function value_info type={elem_type} shape={dims}")
                else:
                    promoted_outputs[out_name] = {"name": out_name, "type": TensorProto.FLOAT, "shape": ["N", "C", "H", "W"]}
                    logging.debug(f"[PROMOTE-FUNC] Promoting function output '{out_name}' with default type/shape")
            else:
                # not produced by any top-level node — cannot promote because save_model expects node outputs
                logging.debug(f"[PROMOTE-FUNC] Skipping function output '{out_name}' (no top-level producer)")

    logging.debug(f"\nPromoted outputs (final): {list(promoted_outputs.keys())}")

    # Final filter of vis
    promoted_output_names = set(promoted_outputs.keys())
    vis = [v for v in vis if v.name not in node_output_names and v.name not in promoted_output_names]
    logging.debug(f"[FILTER] value_info after final cleanup: {[v.name for v in vis]}")

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

