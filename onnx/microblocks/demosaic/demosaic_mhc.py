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

    def __init__(self, version: str = "v0"):
        super().__init__(version=version)

    def build_algo(self, stage, prev_stages=None):
        return super().build_algo(stage, prev_stages)

    def build_applier(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage
        cfa4 = f"{upstream}.applier"  # [n,4,h/2,w/2]
    
        # Registry uses "class" as family identifier
        cfa_onehot = Registry().getInstance().getMapping("bayer2cfa", prev_stages).getParam("cfa_onehot")
    
        out = f"{stage}.applier"      # [n,3,h,w]
        nodes, inits, vis = [], [], []
    
        # Local helper: compute aligned minima and return 4-D sliced tensors
        def _add_dynamic_align(pair_plane, pair_mask, suffix):
            # shapes
            shape_plane = f"{stage}.{suffix}.shape_plane"
            shape_mask = f"{stage}.{suffix}.shape_mask"
            nodes.append(oh.make_node("Shape", [pair_plane], [shape_plane], name=f"{stage}_shape_{suffix}_plane"))
            nodes.append(oh.make_node("Shape", [pair_mask], [shape_mask], name=f"{stage}_shape_{suffix}_mask"))
    
            # gather dims: channel index=1, height=2, width=3
            idx_c_name = f"{stage}.{suffix}.idx_c"
            idx_h_name = f"{stage}.{suffix}.idx_h"
            idx_w_name = f"{stage}.{suffix}.idx_w"
            inits.append(oh.make_tensor(idx_c_name, TensorProto.INT64, [1], [1]))
            inits.append(oh.make_tensor(idx_h_name, TensorProto.INT64, [1], [2]))
            inits.append(oh.make_tensor(idx_w_name, TensorProto.INT64, [1], [3]))
    
            plane_c = f"{stage}.{suffix}.plane_c"
            mask_c = f"{stage}.{suffix}.mask_c"
            plane_h = f"{stage}.{suffix}.plane_h"
            mask_h = f"{stage}.{suffix}.mask_h"
            plane_w = f"{stage}.{suffix}.plane_w"
            mask_w = f"{stage}.{suffix}.mask_w"
    
            nodes.append(oh.make_node("Gather", [shape_plane, idx_c_name], [plane_c], axis=0, name=f"{stage}_gather_{suffix}_plane_c"))
            nodes.append(oh.make_node("Gather", [shape_mask,  idx_c_name], [mask_c],  axis=0, name=f"{stage}_gather_{suffix}_mask_c"))
            nodes.append(oh.make_node("Gather", [shape_plane, idx_h_name], [plane_h], axis=0, name=f"{stage}_gather_{suffix}_plane_h"))
            nodes.append(oh.make_node("Gather", [shape_mask,  idx_h_name], [mask_h],  axis=0, name=f"{stage}_gather_{suffix}_mask_h"))
            nodes.append(oh.make_node("Gather", [shape_plane, idx_w_name], [plane_w], axis=0, name=f"{stage}_gather_{suffix}_plane_w"))
            nodes.append(oh.make_node("Gather", [shape_mask,  idx_w_name], [mask_w],  axis=0, name=f"{stage}_gather_{suffix}_mask_w"))
    
            # compute minima
            min_c = f"{stage}.{suffix}.min_c"
            min_h = f"{stage}.{suffix}.min_h"
            min_w = f"{stage}.{suffix}.min_w"
            nodes.append(oh.make_node("Min", [plane_c, mask_c], [min_c], name=f"{stage}_min_{suffix}_c"))
            nodes.append(oh.make_node("Min", [plane_h, mask_h], [min_h], name=f"{stage}_min_{suffix}_h"))
            nodes.append(oh.make_node("Min", [plane_w, mask_w], [min_w], name=f"{stage}_min_{suffix}_w"))
    
            # --- 64-alignment logic ---
            const64_i = f"{stage}.{suffix}.const64_i"
            const63_i = f"{stage}.{suffix}.const63_i"
            const64_f = f"{stage}.{suffix}.const64_f"
            inits.append(oh.make_tensor(const64_i, TensorProto.INT64, [1], [64]))
            inits.append(oh.make_tensor(const63_i, TensorProto.INT64, [1], [63]))
            inits.append(oh.make_tensor(const64_f, TensorProto.FLOAT, [1], [64.0]))
    
            # cast min_c to float, divide by 64.0, floor, multiply back, cast to int64 -> aligned_down_c
            min_c_f = f"{stage}.{suffix}.min_c_f"
            div_c = f"{stage}.{suffix}.div_c"
            floor_c = f"{stage}.{suffix}.floor_c"
            aligned_down_c_f = f"{stage}.{suffix}.aligned_down_c_f"
            aligned_down_c = f"{stage}.{suffix}.aligned_down_c"
    
            nodes.append(oh.make_node("Cast", [min_c], [min_c_f], to=TensorProto.FLOAT, name=f"{stage}_cast_{suffix}_min_c_f"))
            nodes.append(oh.make_node("Div", [min_c_f, const64_f], [div_c], name=f"{stage}_div_{suffix}_c"))
            nodes.append(oh.make_node("Floor", [div_c], [floor_c], name=f"{stage}_floor_{suffix}_c"))
            nodes.append(oh.make_node("Mul", [floor_c, const64_f], [aligned_down_c_f], name=f"{stage}_mul_{suffix}_c"))
            nodes.append(oh.make_node("Cast", [aligned_down_c_f], [aligned_down_c], to=TensorProto.INT64, name=f"{stage}_cast_{suffix}_aligned_down_c"))
    
            # If min_c < 64, keep min_c; else use aligned_down_c
            cond_c = f"{stage}.{suffix}.cond_c"
            aligned_min_c = f"{stage}.{suffix}.aligned_min_c"
            nodes.append(oh.make_node("Greater", [min_c, const63_i], [cond_c], name=f"{stage}_gt_{suffix}_63"))
            nodes.append(oh.make_node("Where", [cond_c, aligned_down_c, min_c], [aligned_min_c], name=f"{stage}_where_{suffix}_c"))
    
            # Repeat same pattern for height
            min_h_f = f"{stage}.{suffix}.min_h_f"
            div_h = f"{stage}.{suffix}.div_h"
            floor_h = f"{stage}.{suffix}.floor_h"
            aligned_down_h_f = f"{stage}.{suffix}.aligned_down_h_f"
            aligned_down_h = f"{stage}.{suffix}.aligned_down_h"
            nodes.append(oh.make_node("Cast", [min_h], [min_h_f], to=TensorProto.FLOAT, name=f"{stage}_cast_{suffix}_min_h_f"))
            nodes.append(oh.make_node("Div", [min_h_f, const64_f], [div_h], name=f"{stage}_div_{suffix}_h"))
            nodes.append(oh.make_node("Floor", [div_h], [floor_h], name=f"{stage}_floor_{suffix}_h"))
            nodes.append(oh.make_node("Mul", [floor_h, const64_f], [aligned_down_h_f], name=f"{stage}_mul_{suffix}_h"))
            nodes.append(oh.make_node("Cast", [aligned_down_h_f], [aligned_down_h], to=TensorProto.INT64, name=f"{stage}_cast_{suffix}_aligned_down_h"))
            cond_h = f"{stage}.{suffix}.cond_h"
            aligned_min_h = f"{stage}.{suffix}.aligned_min_h"
            nodes.append(oh.make_node("Greater", [min_h, const63_i], [cond_h], name=f"{stage}_gt_{suffix}_63_h"))
            nodes.append(oh.make_node("Where", [cond_h, aligned_down_h, min_h], [aligned_min_h], name=f"{stage}_where_{suffix}_h"))
    
            # Repeat same pattern for width
            min_w_f = f"{stage}.{suffix}.min_w_f"
            div_w = f"{stage}.{suffix}.div_w"
            floor_w = f"{stage}.{suffix}.floor_w"
            aligned_down_w_f = f"{stage}.{suffix}.aligned_down_w_f"
            aligned_down_w = f"{stage}.{suffix}.aligned_down_w"
            nodes.append(oh.make_node("Cast", [min_w], [min_w_f], to=TensorProto.FLOAT, name=f"{stage}_cast_{suffix}_min_w_f"))
            nodes.append(oh.make_node("Div", [min_w_f, const64_f], [div_w], name=f"{stage}_div_{suffix}_w"))
            nodes.append(oh.make_node("Floor", [div_w], [floor_w], name=f"{stage}_floor_{suffix}_w"))
            nodes.append(oh.make_node("Mul", [floor_w, const64_f], [aligned_down_w_f], name=f"{stage}_mul_{suffix}_w"))
            nodes.append(oh.make_node("Cast", [aligned_down_w_f], [aligned_down_w], to=TensorProto.INT64, name=f"{stage}_cast_{suffix}_aligned_down_w"))
            cond_w = f"{stage}.{suffix}.cond_w"
            aligned_min_w = f"{stage}.{suffix}.aligned_min_w"
            nodes.append(oh.make_node("Greater", [min_w, const63_i], [cond_w], name=f"{stage}_gt_{suffix}_63_w"))
            nodes.append(oh.make_node("Where", [cond_w, aligned_down_w, min_w], [aligned_min_w], name=f"{stage}_where_{suffix}_w"))
    
            # --- Ensure ends are 1-D tensors for Slice by reshaping aligned minima to shape [1] ---
            shape1_name = f"{stage}.{suffix}.shape1"
            inits.append(oh.make_tensor(shape1_name, TensorProto.INT64, [1], [1]))
    
            min_c_u = f"{stage}.{suffix}.min_c_u"
            min_h_u = f"{stage}.{suffix}.min_h_u"
            min_w_u = f"{stage}.{suffix}.min_w_u"
            nodes.append(oh.make_node("Reshape", [aligned_min_c, shape1_name], [min_c_u], name=f"{stage}_reshape_{suffix}_c"))
            nodes.append(oh.make_node("Reshape", [aligned_min_h, shape1_name], [min_h_u], name=f"{stage}_reshape_{suffix}_h"))
            nodes.append(oh.make_node("Reshape", [aligned_min_w, shape1_name], [min_w_u], name=f"{stage}_reshape_{suffix}_w"))
    
            # starts zeros for 4-D slice (we'll use a single starts initializer)
            starts4_name = f"{stage}.{suffix}.starts4"
            inits.append(oh.make_tensor(starts4_name, TensorProto.INT64, [4], [0, 0, 0, 0]))
    
            # gather batch dim N from original plane shape (axis 0)
            batch_idx_name = f"{stage}.{suffix}.idx_n"
            inits.append(oh.make_tensor(batch_idx_name, TensorProto.INT64, [1], [0]))
            orig_shape_name = f"{stage}.{suffix}.orig_shape"
            nodes.append(oh.make_node("Shape", [pair_plane], [orig_shape_name], name=f"{stage}_shape_orig_{suffix}"))
            batch_dim = f"{stage}.{suffix}.batch_dim"
            nodes.append(oh.make_node("Gather", [orig_shape_name, batch_idx_name], [batch_dim], axis=0, name=f"{stage}_gather_{suffix}_batch"))
    
            # Make batch_dim 1-D (reshape to [1]) so Concat inputs all have rank 1
            batch_dim_u = f"{stage}.{suffix}.batch_dim_u"
            nodes.append(oh.make_node("Reshape", [batch_dim, shape1_name], [batch_dim_u], name=f"{stage}_reshape_{suffix}_batch_to_1d"))
    
            # concat ends = [batch_dim_u, min_c_u, min_h_u, min_w_u]
            concat_shape_name = f"{stage}.{suffix}.target_shape"
            nodes.append(oh.make_node("Concat", [batch_dim_u, min_c_u, min_h_u, min_w_u], [concat_shape_name], axis=0, name=f"{stage}_concat_{suffix}_target_shape"))
    
            # axes for 4-D slice
            axes4_name = f"{stage}.{suffix}.axes4"
            inits.append(oh.make_tensor(axes4_name, TensorProto.INT64, [4], [0, 1, 2, 3]))
    
            # Single Slice that keeps rank 4: Slice(input, starts=[0,0,0,0], ends=concat_shape_name, axes=[0,1,2,3])
            sliced_4d_plane = f"{pair_plane}_aligned_4d"
            sliced_4d_mask  = f"{pair_mask}_aligned_4d"
            nodes.append(oh.make_node("Slice", [pair_plane, starts4_name, concat_shape_name, axes4_name], [sliced_4d_plane], name=f"{stage}_slice_{suffix}_4d_plane"))
            nodes.append(oh.make_node("Slice", [pair_mask,  starts4_name, concat_shape_name, axes4_name], [sliced_4d_mask],  name=f"{stage}_slice_{suffix}_4d_mask"))
    
            # Return 4-D aligned tensors for downstream ops
            return sliced_4d_plane, sliced_4d_mask
    
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
    
        # 2) Upsample to full resolution (planes)
        roi    = oh.make_tensor(f"{stage}.roi_empty", TensorProto.FLOAT, [0], [])
        scales = oh.make_tensor(f"{stage}.scales",     TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0])
        inits += [roi, scales]
    
        R_full, G0_full, G1_full, B_full = [f"{stage}.{n}" for n in ("R_full","G0_full","G1_full","B_full")]
        G_full_native = f"{stage}.G_full_native"
        # Use CPU-friendly Resize attributes for plane upsample
        nodes += [
            oh.make_node("Resize", [R_half,  f"{stage}.roi_empty", f"{stage}.scales"], [R_full],  name=f"{stage}_resize_R",
                         mode="linear", coordinate_transformation_mode="half_pixel"),
            oh.make_node("Resize", [G0_half, f"{stage}.roi_empty", f"{stage}.scales"], [G0_full], name=f"{stage}_resize_G0",
                         mode="linear", coordinate_transformation_mode="half_pixel"),
            oh.make_node("Resize", [G1_half, f"{stage}.roi_empty", f"{stage}.scales"], [G1_full], name=f"{stage}_resize_G1",
                         mode="linear", coordinate_transformation_mode="half_pixel"),
            oh.make_node("Resize", [B_half,  f"{stage}.roi_empty", f"{stage}.scales"], [B_full],  name=f"{stage}_resize_B",
                         mode="linear", coordinate_transformation_mode="half_pixel"),
            oh.make_node("Add",    [G0_full, G1_full], [G_full_native], name=f"{stage}_sum_G_native"),
        ]
    
        # 3) Upsample one-hot CFA -> masks using grouped ConvTranspose (2x nearest)
        M_stack = f"{stage}.mask_stack"
        M_R, M_G0, M_G1, M_B, M_G = [f"{stage}.{n}" for n in ("M_R","M_G0","M_G1","M_B","M_G")]
    
        # slice indices for channels (after upsample the channel axis is 1 for NCHW)
        (sMR,eMR,aMR),     inits_mR  = _slice_inputs_1d(stage, "mask_R",  starts=[0], ends=[1], axes=[1])
        (sMG0,eMG0,aMG0),  inits_mG0 = _slice_inputs_1d(stage, "mask_G0", starts=[1], ends=[2], axes=[1])
        (sMG1,eMG1,aMG1),  inits_mG1 = _slice_inputs_1d(stage, "mask_G1", starts=[2], ends=[3], axes=[1])
        (sMB,eMB,aMB),     inits_mB  = _slice_inputs_1d(stage, "mask_B",  starts=[3], ends=[4], axes=[1])
        inits += inits_mR + inits_mG0 + inits_mG1 + inits_mB
    
        # Transpose + Cast one-hot to NCHW float for ConvTranspose
        cfa_onehot_nchw = f"{stage}.cfa_onehot_nchw"
        nodes.append(oh.make_node("Transpose", [cfa_onehot], [cfa_onehot_nchw], perm=[0, 3, 1, 2], name=f"{stage}_transpose_onehot"))
    
        cfa_onehot_f = f"{stage}.cfa_onehot_f"
        nodes.append(oh.make_node("Cast", [cfa_onehot_nchw], [cfa_onehot_f], to=TensorProto.FLOAT, name=f"{stage}_cast_onehot_to_float"))
    
        conv_input = cfa_onehot_f  # use the transposed+cast tensor as ConvTranspose input
    
        # Build grouped ConvTranspose kernel for C channels, kernel shape [C,1,2,2]
        C = 4
        kernel_vals = []
        for _ in range(C):
            kernel_vals.extend([1.0, 1.0, 1.0, 1.0])  # 2x2 ones per channel
    
        kernel_name = f"{stage}.upsample_2x_kernel"
        inits.append(oh.make_tensor(kernel_name, TensorProto.FLOAT, [C, 1, 2, 2], kernel_vals))
    
        # Grouped ConvTranspose upsamples by factor 2 (stride=2) and replicates values into 2x2 blocks
        nodes += [
            oh.make_node(
                "ConvTranspose",
                inputs=[conv_input, kernel_name],
                outputs=[M_stack],
                name=f"{stage}_upsample_onehot_ct",
                strides=[2, 2],
                pads=[0, 0, 0, 0],
                group=C
            ),
            # slice channels into per-channel masks
            oh.make_node("Slice", [M_stack, sMR,  eMR,  aMR],  [M_R],  name=f"{stage}_mask_R"),
            oh.make_node("Slice", [M_stack, sMG0, eMG0, aMG0], [M_G0], name=f"{stage}_mask_G0"),
            oh.make_node("Slice", [M_stack, sMG1, eMG1, aMG1], [M_G1], name=f"{stage}_mask_G1"),
            oh.make_node("Slice", [M_stack, sMB,  eMB,  aMB],  [M_B],  name=f"{stage}_mask_B"),
            oh.make_node("Add",   [M_G0, M_G1], [M_G], name=f"{stage}_mask_G"),
        ]
    
        # 4) Kernel initializers for interpolation
        inits += mhc_kernel_inits(stage)
    
        # Add dynamic alignment for R, G, B pairs (returns 4-D aligned tensors)
        R_aligned, M_R_aligned = _add_dynamic_align(R_full, M_R, "R_mask")
        G_aligned, M_G_aligned = _add_dynamic_align(G_full_native, M_G, "G_mask")
        B_aligned, M_B_aligned = _add_dynamic_align(B_full, M_B, "B_mask")
    
        # 5) Cast masks to float (use aligned masks)
        M_R_float = f"{stage}.M_R_float"
        M_B_float = f"{stage}.M_B_float"
        M_G_float = f"{stage}.M_G_float"
    
        nodes += [
            oh.make_node("Cast", [M_R_aligned], [M_R_float], to=TensorProto.FLOAT, name=f"{stage}_cast_M_R"),
            oh.make_node("Cast", [M_B_aligned], [M_B_float], to=TensorProto.FLOAT, name=f"{stage}_cast_M_B"),
            oh.make_node("Cast", [M_G_aligned], [M_G_float], to=TensorProto.FLOAT, name=f"{stage}_cast_M_G"),
        ]
    
        # 6) Green interpolation at R/B sites (use aligned masks and aligned planes)
        G_at_R_conv    = f"{stage}.G_at_R_conv"
        G_at_R_masked  = f"{stage}.G_at_R_masked"
        G_at_B_conv    = f"{stage}.G_at_B_conv"
        G_at_B_masked  = f"{stage}.G_at_B_masked"
        G_interp       = f"{stage}.G_interp"
    
        nodes += [
            oh.make_node("Conv", [R_aligned, f"{stage}.K_mhc_g_r"], [G_at_R_conv],   name=f"{stage}_conv_G_at_R"),
            oh.make_node("Mul",  [G_at_R_conv, M_R_float],       [G_at_R_masked], name=f"{stage}_mask_G_at_R"),
            oh.make_node("Conv", [B_aligned, f"{stage}.K_mhc_g_b"], [G_at_B_conv],   name=f"{stage}_conv_G_at_B"),
            oh.make_node("Mul",  [G_at_B_conv, M_B_float],       [G_at_B_masked], name=f"{stage}_mask_G_at_B"),
            oh.make_node("Add",  [G_at_R_masked, G_at_B_masked], [G_interp],      name=f"{stage}_sum_G_interp"),
        ]
    
        # 7) Red interpolation (aligned)
        R_at_B_conv    = f"{stage}.R_at_B_conv"
        R_at_B_masked  = f"{stage}.R_at_B_masked"
        R_at_G_h_conv  = f"{stage}.R_at_G_h_conv"
        R_at_G_h_masked= f"{stage}.R_at_G_h_masked"
        R_at_G_v_conv  = f"{stage}.R_at_G_v_conv"
        R_at_G_v_masked= f"{stage}.R_at_G_v_masked"
        R_interp_part  = f"{stage}.R_interp_part"
        R_interp_total = f"{stage}.R_interp_total"
    
        nodes += [
            oh.make_node("Conv", [B_aligned, f"{stage}.K_mhc_r_b"],     [R_at_B_conv],     name=f"{stage}_conv_R_at_B"),
            oh.make_node("Mul",  [R_at_B_conv, M_B_float],           [R_at_B_masked],   name=f"{stage}_mask_R_at_B"),
            oh.make_node("Conv", [G_aligned, f"{stage}.K_mhc_r_g_h"], [R_at_G_h_conv], name=f"{stage}_conv_R_at_G_h"),
            oh.make_node("Mul",  [R_at_G_h_conv, M_G_float],         [R_at_G_h_masked], name=f"{stage}_mask_R_at_G_h"),
            oh.make_node("Conv", [G_aligned, f"{stage}.K_mhc_r_g_v"], [R_at_G_v_conv], name=f"{stage}_conv_R_at_G_v"),
            oh.make_node("Mul",  [R_at_G_v_conv, M_G_float],         [R_at_G_v_masked], name=f"{stage}_mask_R_at_G_v"),
            oh.make_node("Add",  [R_at_G_h_masked, R_at_G_v_masked], [R_interp_part],   name=f"{stage}_sum_R_interp"),
            oh.make_node("Add",  [R_interp_part, R_at_B_masked],     [R_interp_total],  name=f"{stage}_sum_R_total"),
        ]
    
        # 8) Blue interpolation (aligned)
        B_at_R_conv    = f"{stage}.B_at_R_conv"
        B_at_R_masked  = f"{stage}.B_at_R_masked"
        B_at_G_h_conv  = f"{stage}.B_at_G_h_conv"
        B_at_G_h_masked= f"{stage}.B_at_G_h_masked"
        B_at_G_v_conv  = f"{stage}.B_at_G_v_conv"
        B_at_G_v_masked= f"{stage}.B_at_G_v_masked"
        B_interp_part  = f"{stage}.B_interp_part"
        B_interp_total = f"{stage}.B_interp_total"
    
        nodes += [
            oh.make_node("Conv", [R_aligned, f"{stage}.K_mhc_b_r"],     [B_at_R_conv],     name=f"{stage}_conv_B_at_R"),
            oh.make_node("Mul",  [B_at_R_conv, M_R_float],           [B_at_R_masked],   name=f"{stage}_mask_B_at_R"),
            oh.make_node("Conv", [G_aligned, f"{stage}.K_mhc_b_g_h"], [B_at_G_h_conv], name=f"{stage}_conv_B_at_G_h"),
            oh.make_node("Mul",  [B_at_G_h_conv, M_G_float],         [B_at_G_h_masked], name=f"{stage}_mask_B_at_G_h"),
            oh.make_node("Conv", [G_aligned, f"{stage}.K_mhc_b_g_v"], [B_at_G_v_conv], name=f"{stage}_conv_B_at_G_v"),
            oh.make_node("Mul",  [B_at_G_v_conv, M_G_float],         [B_at_G_v_masked], name=f"{stage}_mask_B_at_G_v"),
            oh.make_node("Add",  [B_at_G_h_masked, B_at_G_v_masked], [B_interp_part],   name=f"{stage}_sum_B_interp"),
            oh.make_node("Add",  [B_interp_part, B_at_R_masked],     [B_interp_total],  name=f"{stage}_sum_B_total"),
        ]
    
        # 9) Blend native + interpolated -> RGB
        R_native, G_native, B_native = [f"{stage}.{n}" for n in ("R_native","G_native","B_native")]
        R_out, G_out, B_out = [f"{stage}.{n}" for n in ("R_out","G_out","B_out")]
        nodes += [
            oh.make_node("Mul", [R_aligned,        M_R_float], [R_native], name=f"{stage}_native_R"),
            oh.make_node("Add", [R_native,      R_interp_total], [R_out], name=f"{stage}_out_R"),
    
            oh.make_node("Mul", [G_aligned, M_G_float], [G_native], name=f"{stage}_native_G"),
            oh.make_node("Add", [G_native,      G_interp], [G_out],     name=f"{stage}_out_G"),
    
            oh.make_node("Mul", [B_aligned,        M_B_float], [B_native], name=f"{stage}_native_B"),
            oh.make_node("Add", [B_native,      B_interp_total], [B_out], name=f"{stage}_out_B"),
    
            oh.make_node("Concat", [R_out, G_out, B_out], [out], axis=1, name=f"{stage}_concat_RGB"),
        ]
    
        # Value info: ensure shapes are consistent and authoritative
        vis += [
            oh.make_tensor_value_info(cfa4,       TensorProto.FLOAT, ["n",4,"h","w"]),
            oh.make_tensor_value_info(cfa_onehot, TensorProto.FLOAT, [1,2,2,4]),
            oh.make_tensor_value_info(M_stack,    TensorProto.FLOAT, ["n",4,"h","w"]),
            oh.make_tensor_value_info(out,        TensorProto.FLOAT, ["n",3,"h","w"]),
        ]
    
        outputs = {"applier": {"name": out, "type":TensorProto.FLOAT, "shape":["n",3,"h","w"]}}
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(cfa4, type=TensorProto.FLOAT, shape=["n",4,"h","w"])        # CFA planes
        result.appendInput(cfa_onehot, type=TensorProto.FLOAT, shape=[1,2,2,4])  # one‑hot tile
        return result
