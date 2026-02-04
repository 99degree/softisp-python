from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.wb.wb_base import AWBBase


class WBAvgV1(AWBBase):
    """
    WBAvgV1 (v1)
    -------------
    Direct RGB averaging white balance block.

    Inputs (external):
        - prev_stage.applier : upstream image tensor [n,3,h,w]

    Outputs:
        - wb_avg_v1.applier  : WB-applied image [n,3,h,w]
        - wb_avg_v1.wb_gains : estimated WB gains [1,3,1,1]
        - wb_avg_v1.cct      : correlated color temperature [1]
    """
    name = "wb_avg_v1"
    version = "v1"

    # -------------------------------
    # Trunk 1 — Channel means (R, G, B)
    # -------------------------------
    def _build_channel_means(self, stage, prev_stage, nodes, inits, vis):
        """
        Compute mean R,G,B from upstream stage image [n,3,h,w].
        """
        in_image = f"{prev_stage}.applier"
        mean_name = f"{stage}.mean_channels"

        # ReduceMean over spatial dims -> [n,3]
        nodes.append(
            oh.make_node(
                "ReduceMean",
                inputs=[in_image],
                outputs=[mean_name],
                name=f"{stage}.reduce_mean",
                keepdims=0,
                axes=[2, 3],
            )
        )

        # Gather channel means separately
        r_mean, g_mean, b_mean = (
            f"{stage}.r_mean",
            f"{stage}.g_mean",
            f"{stage}.b_mean",
        )
        inits.append(oh.make_tensor(f"{stage}.r_index", TensorProto.INT64, [1], [0]))
        inits.append(oh.make_tensor(f"{stage}.g_index", TensorProto.INT64, [1], [1]))
        inits.append(oh.make_tensor(f"{stage}.b_index", TensorProto.INT64, [1], [2]))

        nodes.append(oh.make_node("Gather", [mean_name, f"{stage}.r_index"], [r_mean], axis=1))
        nodes.append(oh.make_node("Gather", [mean_name, f"{stage}.g_index"], [g_mean], axis=1))
        nodes.append(oh.make_node("Gather", [mean_name, f"{stage}.b_index"], [b_mean], axis=1))

        g_mean_avg = g_mean  # if RGGB, average G1/G2; here single G

        vis += [
            oh.make_tensor_value_info(in_image, TensorProto.FLOAT, ["n", 3, "h", "w"]),
            oh.make_tensor_value_info(mean_name, TensorProto.FLOAT, ["n", 3]),
            oh.make_tensor_value_info(r_mean, TensorProto.FLOAT, ["n", 1]),
            oh.make_tensor_value_info(g_mean_avg, TensorProto.FLOAT, ["n", 1]),
            oh.make_tensor_value_info(b_mean, TensorProto.FLOAT, ["n", 1]),
        ]

        return r_mean, g_mean_avg, b_mean

    # -------------------------------
    # Trunk 2 — WB gains + CCT
    # -------------------------------
    def _build_wb_gains(self, stage, nodes, inits, r_mean, g_mean_avg, b_mean):
        """
        Compute WB gains from channel means.
        Output: [n,3,1,1] tensor (broadcastable over [n,3,h,w])
        """
        gain_r, gain_g, gain_b = f"{stage}.gain_r", f"{stage}.gain_g", f"{stage}.gain_b"
        out_wb_vec = f"{stage}.wb_gains_vec"
        out_wb     = f"{stage}.wb_gains"

        # Compute per-batch gains (each [n,1])
        nodes.append(oh.make_node("Div", [g_mean_avg, r_mean], [gain_r]))
        nodes.append(oh.make_node("Div", [g_mean_avg, g_mean_avg], [gain_g]))
        nodes.append(oh.make_node("Div", [g_mean_avg, b_mean], [gain_b]))

        # Concat along axis=1 → [n,3]
        nodes.append(
            oh.make_node("Concat", [gain_r, gain_g, gain_b],
                         [out_wb_vec], axis=1)
        )

        # Reshape [n,3] → [n,3,1,1]
        shape_const = f"{stage}.wb_shape"
        inits.append(oh.make_tensor(shape_const, TensorProto.INT64, [4], [-1, 3, 1, 1]))
        nodes.append(oh.make_node("Reshape", [out_wb_vec, shape_const], [out_wb]))

        return out_wb

    def _build_cct(self, stage, nodes, inits, r_mean, g_mean_avg, b_mean):
        """
        Estimate CCT from chromaticities using McCamy’s formula.
        """
        out_cct = f"{stage}.cct_fix"

        # Chromaticity sums
        sum_rg, sum_rgb = f"{stage}.sum_rg", f"{stage}.sum_rgb"
        nodes.append(oh.make_node("Add", [r_mean, g_mean_avg], [sum_rg], name=f"{stage}.sum_rg"))
        nodes.append(oh.make_node("Add", [sum_rg, b_mean], [sum_rgb], name=f"{stage}.sum_rgb"))

        # Chromaticities
        r_chroma, b_chroma = f"{stage}.r_chroma", f"{stage}.b_chroma"
        nodes.append(oh.make_node("Div", [r_mean, sum_rgb], [r_chroma], name=f"{stage}.r_chroma"))
        nodes.append(oh.make_node("Div", [b_mean, sum_rgb], [b_chroma], name=f"{stage}.b_chroma"))

        # Constants
        const_r, const_b = f"{stage}.const_r", f"{stage}.const_b"
        inits.append(oh.make_tensor(const_r, TensorProto.FLOAT, [], [0.332]))
        inits.append(oh.make_tensor(const_b, TensorProto.FLOAT, [], [0.1858]))

        # Shifts
        r_shift, b_shift = f"{stage}.r_shift", f"{stage}.b_shift"
        nodes.append(oh.make_node("Sub", [r_chroma, const_r], [r_shift], name=f"{stage}.r_shift"))
        nodes.append(oh.make_node("Sub", [b_chroma, const_b], [b_shift], name=f"{stage}.b_shift"))

        # Polynomial terms
        n_ratio, n2, n3 = f"{stage}.n_ratio", f"{stage}.n2", f"{stage}.n3"
        nodes.append(oh.make_node("Div", [r_shift, b_shift], [n_ratio], name=f"{stage}.n_ratio"))
        nodes.append(oh.make_node("Mul", [n_ratio, n_ratio], [n2], name=f"{stage}.n2"))
        nodes.append(oh.make_node("Mul", [n2, n_ratio], [n3], name=f"{stage}.n3"))

        # McCamy coefficients
        coeffs = {"c3": -449.0, "c2": 3525.0, "c1": -6823.3, "c0": 5520.33}
        for cname, val in coeffs.items():
            const = f"{stage}.{cname}_const"
            inits.append(oh.make_tensor(const, TensorProto.FLOAT, [], [val]))

        # Polynomial evaluation
        term3, term2, term1 = f"{stage}.term3", f"{stage}.term2", f"{stage}.term1"
        nodes.append(oh.make_node("Mul", [n3, f"{stage}.c3_const"], [term3], name=f"{stage}.term3"))
        nodes.append(oh.make_node("Mul", [n2, f"{stage}.c2_const"], [term2], name=f"{stage}.term2"))
        nodes.append(oh.make_node("Mul", [n_ratio, f"{stage}.c1_const"], [term1], name=f"{stage}.term1"))

        tmp_sum1, tmp_sum2 = f"{stage}.tmp_sum1", f"{stage}.tmp_sum2"
        nodes.append(oh.make_node("Add", [term3, term2], [tmp_sum1], name=f"{stage}.sum_terms1"))
        nodes.append(oh.make_node("Add", [tmp_sum1, term1], [tmp_sum2], name=f"{stage}.sum_terms2"))
        nodes.append(oh.make_node("Add", [tmp_sum2, f"{stage}.c0_const"], [out_cct], name=f"{stage}.add_c0"))

        return out_cct

    # -------------------------------
    # Trunk 3 — wb_gain application
    # -------------------------------
    def _build_wb_gain_apply(self, stage, prev_stage, nodes, inits, vis):
        """
        Apply wb_gain to upstream image [n,3,H,W] with broadcasted [1,3,1,1].
        """
        input_image = f"{prev_stage}.applier"
        wb_gain_in  = f"{stage}.wb_gains"
        shape_name  = f"{stage}.image_shape"
        wb_gain_exp = f"{stage}.wb_gains_expanded"
        applier     = f"{stage}.applier"

        # Shape of input image
        nodes.append(oh.make_node("Shape", [input_image], [shape_name], name=f"{stage}_shape"))

        # Expand gains to match image shape
        nodes.append(oh.make_node("Expand", [wb_gain_in, shape_name], [wb_gain_exp], name=f"{stage}_expand_gains"))

        # Multiply
        nodes.append(oh.make_node("Mul", [input_image, wb_gain_exp], [applier], name=f"{stage}_wb_gain"))

        vis += [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["N", 3, "H", "W"]),
            oh.make_tensor_value_info(wb_gain_in,  TensorProto.FLOAT, [1, 3, 1, 1]),
            oh.make_tensor_value_info(wb_gain_exp, TensorProto.FLOAT, ["N", 3, "H", "W"]),
            oh.make_tensor_value_info(applier,     TensorProto.FLOAT, ["N", 3, "H", "W"]),
        ]

        return applier, input_image

    def _fix_cct_shape(self, stage, nodes, inits, vis, in_cct):
        """
        Ensure CCT is a rank-1 vector [1] instead of [1,1].
        """
        out_cct = f"{stage}.cct"
        shape_const = f"{stage}.cct_shape"
        inits.append(oh.make_tensor(shape_const, TensorProto.INT64, [1], [1]))
        nodes.append(oh.make_node("Reshape", [in_cct, shape_const], [out_cct], name=f"{stage}_fix_cct"))
        vis.append(oh.make_tensor_value_info(out_cct, TensorProto.FLOAT, [1]))
        return out_cct


    # -------------------------------
    # Stitch trunks — build_algo
    # -------------------------------
    def build_algo(self, stage: str, prev_stages=None):
        nodes, inits, vis = [], [], []
        upstream = prev_stages[0] if prev_stages else stage

        # Trunk 1: channel means
        r_mean, g_mean_avg, b_mean = self._build_channel_means(stage, upstream, nodes, inits, vis)

        # Trunk 2: WB gains + CCT
        out_wb = self._build_wb_gains(stage, nodes, inits, r_mean, g_mean_avg, b_mean)
        raw_cct = self._build_cct(stage, nodes, inits, r_mean, g_mean_avg, b_mean)
        out_cct = self._fix_cct_shape(stage, nodes, inits, vis, raw_cct)

        vis += [
            oh.make_tensor_value_info(out_wb, TensorProto.FLOAT, [1, 3, 1, 1]),
            oh.make_tensor_value_info(out_cct, TensorProto.FLOAT, [1]),
        ]

        # Trunk 3: wb_gain application
        applier, input_image = self._build_wb_gain_apply(stage, upstream, nodes, inits, vis)

        # Outputs
        outputs = {
            "applier": {"name": applier, "type":TensorProto.FLOAT, "shape":[1,3,"h","w"]},
            "wb_gains": {"name": out_wb, "type":TensorProto.FLOAT, "shape":[1, 3, 1, 1]},
            "cct": {"name": out_cct, "type":TensorProto.FLOAT, "shape":[1]},
        }

        # Explicit external inputs for checker-safe wiring
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image,type=TensorProto.FLOAT, shape=[1,3,"h","w"])  # dependent input from upstream
        return result

    def build_applier(self, stage: str, prev_stages=None):
        return super().build_applier(stage, prev_stages)
