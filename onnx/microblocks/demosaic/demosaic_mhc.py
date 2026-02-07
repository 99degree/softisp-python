# demosaic_mhc.py
from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from .demosaic_base import DemosaicBase
from microblocks.registry import Registry

def _kernel_init(stage, name, weights_2d):
    flat = [w for row in weights_2d for w in row]
    return oh.make_tensor(f"{stage}.{name}", TensorProto.FLOAT, [1,1,5,5], flat)

def mhc_kernel_inits(stage):
    return [
        _kernel_init(stage, "K_mhc_g_r",  [[0,0,-1,0,0],[0,0,2,0,0],[-1,2,4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]]),
        _kernel_init(stage, "K_mhc_g_b",  [[0,0,-1,0,0],[0,0,2,0,0],[-1,2,4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]]),
        _kernel_init(stage, "K_mhc_r_b",  [[0,0,-1,0,0],[0,0,2,0,0],[-1,2,5,2,-1],[0,0,2,0,0],[0,0,-1,0,0]]),
        _kernel_init(stage, "K_mhc_b_r",  [[0,0,-1,0,0],[0,0,2,0,0],[-1,2,5,2,-1],[0,0,2,0,0],[0,0,-1,0,0]]),
        _kernel_init(stage, "K_mhc_r_g_h",[[0,0,0,0,0],[0,0,-1,0,0],[0,-1,4,-1,0],[0,0,-1,0,0],[0,0,0,0,0]]),
        _kernel_init(stage, "K_mhc_r_g_v",[[0,0,0,0,0],[0,0,-1,0,0],[0,-1,4,-1,0],[0,0,-1,0,0],[0,0,0,0,0]]),
        _kernel_init(stage, "K_mhc_b_g_h",[[0,0,0,0,0],[0,0,-1,0,0],[0,-1,4,-1,0],[0,0,-1,0,0],[0,0,0,0,0]]),
        _kernel_init(stage, "K_mhc_b_g_v",[[0,0,0,0,0],[0,0,-1,0,0],[0,-1,4,-1,0],[0,0,-1,0,0],[0,0,0,0,0]]),
    ]

def _slice_inputs_1d(stage, tag, starts, ends, axes):
    s_name = f"{stage}.slice_{tag}_starts"
    e_name = f"{stage}.slice_{tag}_ends"
    a_name = f"{stage}.slice_{tag}_axes"
    inits = [
        oh.make_tensor(s_name, TensorProto.INT64, [len(starts)], starts),
        oh.make_tensor(e_name, TensorProto.INT64, [len(ends)],   ends),
        oh.make_tensor(a_name, TensorProto.INT64, [len(axes)],   axes),
    ]
    return (s_name, e_name, a_name), inits

def _tile_repeats_init(stage, repeats):
    name = f"{stage}.tile_repeats"
    return name, oh.make_tensor(name, TensorProto.INT64, [len(repeats)], repeats)

class DemosaicMHC(DemosaicBase):
    name = "demosaic_mhc"
    version = "v0"
    provides = ["applier"]

    def build_algo(self, stage, prev_stages=None):
        return super().build_algo(stage, prev_stages)

    def build_applier(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage
        cfa4 = f"{upstream}.applier"  # [n,4,h/2,w/2]

        # Registry uses "class" as family identifier
        cfa_onehot = Registry().getInstance().getMapping("bayer2cfa", prev_stages).getParam("cfa_onehot")

        out = f"{stage}.applier"      # [n,3,h,w]
        nodes, inits, vis = [], [], []

        # 1) Slice CFA planes
        R_half, G0_half, G1_half, B_half = [f"{stage}.{n}" for n in ("R_half","G0_half","G1_half","B_half")]
        (sR,eR,aR),   inits_R   = _slice_inputs_1d(stage, "R",  starts=[0], ends=[1], axes=[1])
        (sG0,eG0,aG0), inits_G0 = _slice_inputs_1d(stage, "G0", starts=[1], ends=[2], axes=[1])
        (sG1,eG1,aG1), inits_G1 = _slice_inputs_1d(stage, "G1", starts=[2], ends=[3], axes=[1])
        (sB,eB,aB),   inits_B   = _slice_inputs_1d(stage, "B",  starts=[3], ends=[4], axes=[1])
        inits += inits_R + inits_G0 + inits_G1 + inits_B

        nodes += [
            oh.make_node("Slice", [cfa4, sR,  eR,  aR],  [R_half],  name=f"{stage}_slice_R"),
            oh.make_node("Slice", [cfa4, sG0, eG0, aG0], [G0_half], name=f"{stage}_slice_G0"),
            oh.make_node("Slice", [cfa4, sG1, eG1, aG1], [G1_half], name=f"{stage}_slice_G1"),
            oh.make_node("Slice", [cfa4, sB,  eB,  aB],  [B_half],  name=f"{stage}_slice_B"),
        ]

        # 2) Upsample to full resolution
        roi    = oh.make_tensor(f"{stage}.roi_empty", TensorProto.FLOAT, [0], [])
        scales = oh.make_tensor(f"{stage}.scales",     TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0])
        inits += [roi, scales]

        R_full, G0_full, G1_full, B_full = [f"{stage}.{n}" for n in ("R_full","G0_full","G1_full","B_full")]
        G_full_native = f"{stage}.G_full_native"
        nodes += [
            oh.make_node("Resize", [R_half,  f"{stage}.roi_empty", f"{stage}.scales"], [R_full],  name=f"{stage}_resize_R"),
            oh.make_node("Resize", [G0_half, f"{stage}.roi_empty", f"{stage}.scales"], [G0_full], name=f"{stage}_resize_G0"),
            oh.make_node("Resize", [G1_half, f"{stage}.roi_empty", f"{stage}.scales"], [G1_full], name=f"{stage}_resize_G1"),
            oh.make_node("Resize", [B_half,  f"{stage}.roi_empty", f"{stage}.scales"], [B_full],  name=f"{stage}_resize_B"),
            oh.make_node("Add",    [G0_full, G1_full], [G_full_native], name=f"{stage}_sum_G_native"),
        ]

        # 3) Upsample one-hot CFA → masks using Resize (nearest) so masks match plane size exactly
        M_stack = f"{stage}.mask_stack"
        M_R, M_G0, M_G1, M_B, M_G = [f"{stage}.{n}" for n in ("M_R","M_G0","M_G1","M_B","M_G")]

        # slice indices for channels (after Resize the channel axis is 1 for NCHW)
        (sMR,eMR,aMR),     inits_mR  = _slice_inputs_1d(stage, "mask_R",  starts=[0], ends=[1], axes=[1])
        (sMG0,eMG0,aMG0),  inits_mG0 = _slice_inputs_1d(stage, "mask_G0", starts=[1], ends=[2], axes=[1])
        (sMG1,eMG1,aMG1),  inits_mG1 = _slice_inputs_1d(stage, "mask_G1", starts=[2], ends=[3], axes=[1])
        (sMB,eMB,aMB),     inits_mB  = _slice_inputs_1d(stage, "mask_B",  starts=[3], ends=[4], axes=[1])
        inits += inits_mR + inits_mG0 + inits_mG1 + inits_mB

        # If cfa_onehot is NHWC (N,Ht,Wt,4), transpose it to NCHW first:
        # cfa_onehot_nchw = f"{stage}.cfa_onehot_nchw"
        # nodes += [ oh.make_node("Transpose", [cfa_onehot], [cfa_onehot_nchw], perm=[0,3,1,2], name=f"{stage}_transpose_onehot") ]
        # use cfa_onehot_nchw below if you uncomment the transpose

        # Reuse the same scales initializer used for plane Resize (ensure name matches)
        # We create a dedicated roi/scales for masks to avoid name collisions
        roi_masks = oh.make_tensor(f"{stage}.roi_empty_masks", TensorProto.FLOAT, [0], [])
        scales_masks = oh.make_tensor(f"{stage}.scales_masks", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0])
        inits += [roi_masks, scales_masks]

        # Upsample one-hot to full resolution with nearest neighbor so one-hot values remain integer-like
        nodes += [
                oh.make_node(
                        "Resize",
                        [cfa_onehot, f"{stage}.roi_empty_masks", f"{stage}.scales_masks"],
                        [M_stack],
                        name=f"{stage}_resize_onehot",
                        mode="nearest",
                        coordinate_transformation_mode="asymmetric",
                        nearest_mode="floor"
                ),
                # slice channels into per-channel masks
                oh.make_node("Slice", [M_stack, sMR,  eMR,  aMR],  [M_R],  name=f"{stage}_mask_R"),
                oh.make_node("Slice", [M_stack, sMG0, eMG0, aMG0], [M_G0], name=f"{stage}_mask_G0"),
                oh.make_node("Slice", [M_stack, sMG1, eMG1, aMG1], [M_G1], name=f"{stage}_mask_G1"),
                oh.make_node("Slice", [M_stack, sMB,  eMB,  aMB],  [M_B],  name=f"{stage}_mask_B"),
                oh.make_node("Add",   [M_G0, M_G1], [M_G], name=f"{stage}_mask_G"),
        ]

        # 4) Kernel initializers
        inits += mhc_kernel_inits(stage)

        # Cast masks to float
        M_R_float = f"{stage}.M_R_float"
        M_B_float = f"{stage}.M_B_float"
        M_G_float = f"{stage}.M_G_float"

        nodes += [
            oh.make_node("Cast", [M_R], [M_R_float], to=TensorProto.FLOAT, name=f"{stage}_cast_M_R"),
            oh.make_node("Cast", [M_B], [M_B_float], to=TensorProto.FLOAT, name=f"{stage}_cast_M_B"),
            oh.make_node("Cast", [M_G], [M_G_float], to=TensorProto.FLOAT, name=f"{stage}_cast_M_G"),
        ]

        # 5) Green interpolation at R/B sites
        G_at_R_conv    = f"{stage}.G_at_R_conv"
        G_at_R_masked  = f"{stage}.G_at_R_masked"
        G_at_B_conv    = f"{stage}.G_at_B_conv"
        G_at_B_masked  = f"{stage}.G_at_B_masked"
        G_interp       = f"{stage}.G_interp"

        nodes += [
            oh.make_node("Conv", [R_full, f"{stage}.K_mhc_g_r"], [G_at_R_conv],   name=f"{stage}_conv_G_at_R"),
            oh.make_node("Mul",  [G_at_R_conv, M_R_float],       [G_at_R_masked], name=f"{stage}_mask_G_at_R"),
            oh.make_node("Conv", [B_full, f"{stage}.K_mhc_g_b"], [G_at_B_conv],   name=f"{stage}_conv_G_at_B"),
            oh.make_node("Mul",  [G_at_B_conv, M_B_float],       [G_at_B_masked], name=f"{stage}_mask_G_at_B"),
            oh.make_node("Add",  [G_at_R_masked, G_at_B_masked], [G_interp],      name=f"{stage}_sum_G_interp"),
        ]

        # 6) Red interpolation
        R_at_B_conv    = f"{stage}.R_at_B_conv"
        R_at_B_masked  = f"{stage}.R_at_B_masked"
        R_at_G_h_conv  = f"{stage}.R_at_G_h_conv"
        R_at_G_h_masked= f"{stage}.R_at_G_h_masked"
        R_at_G_v_conv  = f"{stage}.R_at_G_v_conv"
        R_at_G_v_masked= f"{stage}.R_at_G_v_masked"
        R_interp_part  = f"{stage}.R_interp_part"
        R_interp_total = f"{stage}.R_interp_total"

        nodes += [
            oh.make_node("Conv", [B_full, f"{stage}.K_mhc_r_b"],     [R_at_B_conv],     name=f"{stage}_conv_R_at_B"),
            oh.make_node("Mul",  [R_at_B_conv, M_B_float],           [R_at_B_masked],   name=f"{stage}_mask_R_at_B"),
            oh.make_node("Conv", [G_full_native, f"{stage}.K_mhc_r_g_h"], [R_at_G_h_conv], name=f"{stage}_conv_R_at_G_h"),
            oh.make_node("Mul",  [R_at_G_h_conv, M_G_float],         [R_at_G_h_masked], name=f"{stage}_mask_R_at_G_h"),
            oh.make_node("Conv", [G_full_native, f"{stage}.K_mhc_r_g_v"], [R_at_G_v_conv], name=f"{stage}_conv_R_at_G_v"),
            oh.make_node("Mul",  [R_at_G_v_conv, M_G_float],         [R_at_G_v_masked], name=f"{stage}_mask_R_at_G_v"),
            oh.make_node("Add",  [R_at_G_h_masked, R_at_G_v_masked], [R_interp_part],   name=f"{stage}_sum_R_interp"),
            oh.make_node("Add",  [R_interp_part, R_at_B_masked],     [R_interp_total],  name=f"{stage}_sum_R_total"),
        ]

        # 7) Blue interpolation
        B_at_R_conv    = f"{stage}.B_at_R_conv"
        B_at_R_masked  = f"{stage}.B_at_R_masked"
        B_at_G_h_conv  = f"{stage}.B_at_G_h_conv"
        B_at_G_h_masked= f"{stage}.B_at_G_h_masked"
        B_at_G_v_conv  = f"{stage}.B_at_G_v_conv"
        B_at_G_v_masked= f"{stage}.B_at_G_v_masked"
        B_interp_part  = f"{stage}.B_interp_part"
        B_interp_total = f"{stage}.B_interp_total"

        nodes += [
            oh.make_node("Conv", [R_full, f"{stage}.K_mhc_b_r"],     [B_at_R_conv],     name=f"{stage}_conv_B_at_R"),
            oh.make_node("Mul",  [B_at_R_conv, M_R_float],           [B_at_R_masked],   name=f"{stage}_mask_B_at_R"),
            oh.make_node("Conv", [G_full_native, f"{stage}.K_mhc_b_g_h"], [B_at_G_h_conv], name=f"{stage}_conv_B_at_G_h"),
            oh.make_node("Mul",  [B_at_G_h_conv, M_G_float],         [B_at_G_h_masked], name=f"{stage}_mask_B_at_G_h"),
            oh.make_node("Conv", [G_full_native, f"{stage}.K_mhc_b_g_v"], [B_at_G_v_conv], name=f"{stage}_conv_B_at_G_v"),
            oh.make_node("Mul",  [B_at_G_v_conv, M_G_float],         [B_at_G_v_masked], name=f"{stage}_mask_B_at_G_v"),
            oh.make_node("Add",  [B_at_G_h_masked, B_at_G_v_masked], [B_interp_part],   name=f"{stage}_sum_B_interp"),
            oh.make_node("Add",  [B_interp_part, B_at_R_masked],     [B_interp_total],  name=f"{stage}_sum_B_total"),
        ]

        # 8) Blend native + interpolated → RGB
        R_native, G_native, B_native = [f"{stage}.{n}" for n in ("R_native","G_native","B_native")]
        R_out, G_out, B_out = [f"{stage}.{n}" for n in ("R_out","G_out","B_out")]
        nodes += [
            oh.make_node("Mul", [R_full,        M_R_float], [R_native], name=f"{stage}_native_R"),
            oh.make_node("Add", [R_native,      R_interp_total], [R_out], name=f"{stage}_out_R"),

            oh.make_node("Mul", [G_full_native, M_G_float], [G_native], name=f"{stage}_native_G"),
            oh.make_node("Add", [G_native,      G_interp], [G_out],     name=f"{stage}_out_G"),

            oh.make_node("Mul", [B_full,        M_B_float], [B_native], name=f"{stage}_native_B"),
            oh.make_node("Add", [B_native,      B_interp_total], [B_out], name=f"{stage}_out_B"),

            oh.make_node("Concat", [R_out, G_out, B_out], [out], axis=1, name=f"{stage}_concat_RGB"),
        ]

        vis += [
            oh.make_tensor_value_info(cfa4,       TensorProto.FLOAT, ["n",4,"h","w"]),
            oh.make_tensor_value_info(cfa_onehot, TensorProto.FLOAT, [1,2,2,4]),
            oh.make_tensor_value_info(out,        TensorProto.FLOAT, ["n",3,"h","w"]),
        ]

        outputs = {"applier": {"name": out, "type":TensorProto.FLOAT, "shape":["n",3,"h","w"]}}
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(cfa4, type=TensorProto.FLOAT, shape=["n",4,"h","w"])        # CFA planes
        result.appendInput(cfa_onehot, type=TensorProto.FLOAT, shape=[1,2,2,4])  # one‑hot tile
        return result
