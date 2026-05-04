from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class LensLCSBase(MicroblockBase):
    """
    LensLCSBase (v0)
    ----------------
    Canonical base microblock for Lens Correction & Shading (LCS).

    Needs:
        - applier [n,3,h,w] : image tensor from upstream
        - lcs_coeffs [h,w]  : per-pixel correction coefficients

    Provides:
        - applier [n,3,h,w] : corrected image tensor

    Behavior:
        - build_algo: declares lcs_coeffs as an external need (no generation here)
        - build_applier: multiplies applier × lcs_coeffs (broadcasted across channels)
    """
    name = 'lens_lcs_base'
    family = 'lcs_lcs_base'
    version = 'v0'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Base version does not generate lcs_coeffs.
        Assumes lcs_coeffs are provided externally (from coordinator or calibration).
        """
        vis, nodes, inits = ([], [], [])
        coeffs_name = f'{stage}.lcs_coeffs'
        vis.append(oh.make_tensor_value_info(coeffs_name, TensorProto.FLOAT, ['h', 'w']))
        outputs = {'lcs_coeffs': {'name': coeffs_name}}
        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{prev_stages[0]}.applier')

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply lens correction coefficients to applier (image).
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        lcs_coeffs = f'{upstream}.lcs_coeffs'
        applier = f'{stage}.applier'
        nodes.append(oh.make_node('Mul', inputs=[input_image, lcs_coeffs], outputs=[applier], name=f'{stage}.mul_apply'))
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        outputs = {'applier': {'name': applier}}
        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{prev_stages[0]}.applier')

    def build_coordinator(self, stage: str, prev_stages=None):
        return super().build_coordinator(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)


class LensLCSV1(MicroblockBase):
    """
    LensLCSV1 (v1)
    --------------
    Adaptive lens shading correction block with coefficient resizing.

    Inputs (external):
        - prev_stage.applier : upstream image tensor [n,3,h,w]
        - lcs_coeffs : full-resolution correction coefficients [H,W]

    Outputs:
        - applier : corrected image tensor [n,3,h*,w*]
        - lcs_coeffs_resized : resized coefficient map [h*,w*]
        - lcs_coeffs_out : identity copy of original coeffs [H,W]
    """
    name = "lens_lcs_v1"
    family = "lcs_lcs_v1"
    version = "v1"

    def build_algo(self, stage: str, prev_stages=None):
        vis, nodes, inits = [], [], []
        upstream = prev_stages[0] if prev_stages else stage

        input_image = f"{upstream}.applier"
        lcs_coeffs  = f"{stage}.lcs_coeffs"

        # Constants as rank-1 tensors
        one_n, one_c = f"{stage}.one_n", f"{stage}.one_c"
        h_factor, w_factor = f"{stage}.h_factor", f"{stage}.w_factor"
        inits += [
            oh.make_tensor(one_n, TensorProto.FLOAT, [1], [1.0]),
            oh.make_tensor(one_c, TensorProto.FLOAT, [1], [1.0]),
            oh.make_tensor(h_factor, TensorProto.FLOAT, [1], [0.5]),
            oh.make_tensor(w_factor, TensorProto.FLOAT, [1], [0.5]),
        ]

        # Concat into [1,1,0.5,0.5]
        scales = f"{stage}.scales"
        nodes.append(
            oh.make_node(
                "Concat",
                inputs=[one_n, one_c, h_factor, w_factor],
                outputs=[scales],
                name=f"{stage}.concat_scales",
                axis=0,
            )
        )
        vis.append(oh.make_tensor_value_info(scales, TensorProto.FLOAT, [4]))

        roi = f"{stage}.roi_empty"
        inits.append(oh.make_tensor(roi, TensorProto.FLOAT, [0], []))

        # Resize using scales only (no sizes)
        lcs_resized = f"{stage}.lcs_coeffs_resized"
        nodes.append(
            oh.make_node(
                "Resize",
                inputs=[lcs_coeffs, roi, scales],  # only data + scales
                outputs=[lcs_resized],
                name=f"{stage}.resize_lcs",
                mode="linear",
            )
        )
        vis.append(oh.make_tensor_value_info(lcs_resized, TensorProto.FLOAT, [1, 1, "h*", "w*"]))

        # Identity for original coeffs
        lcs_coeffs_out = f"{stage}.lcs_coeffs_out"
        nodes.append(
            oh.make_node("Identity", inputs=[lcs_coeffs], outputs=[lcs_coeffs_out], name=f"{stage}.identity_lcs")
        )
        vis.append(oh.make_tensor_value_info(lcs_coeffs_out, TensorProto.FLOAT, [1,1, "H", "W"]))

        # Apply correction
        applier = f"{stage}.applier"
        nodes.append(
            oh.make_node("Mul", inputs=[input_image, lcs_resized], outputs=[applier], name=f"{stage}.algo_mul_apply")
        )
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ["n", 3, "h*", "w*"]))

        outputs = {
            "applier": {"name": applier},
            "lcs_coeffs_resized": {"name": lcs_resized},
            "lcs_coeffs": {"name": lcs_coeffs_out},
        }

        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, ['n','c',"h","w"], type=TensorProto.FLOAT)
        result.appendInput(lcs_coeffs, [1, 1, 'h', 'w'], type=TensorProto.FLOAT)
        return result

    def build_applier(self, stage: str, prev_stages=None):
        vis, nodes, inits = [], [], []
        upstream = prev_stages[0] if prev_stages else stage

        input_image = f"{upstream}.applier"
        lcs_coeffs  = f"{stage}.lcs_coeffs"
        applier     = f"{stage}.applier"

        # Apply correction directly
        nodes.append(
            oh.make_node("Mul", inputs=[input_image, lcs_coeffs], outputs=[applier], name=f"{stage}.mul_apply")
        )

        vis += [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", 3, "h*", "w*"]),
            oh.make_tensor_value_info(lcs_coeffs, TensorProto.FLOAT, [1, 1, "h", "w"]),
            oh.make_tensor_value_info(applier, TensorProto.FLOAT, ["n", 3, "h*", "w*"]),
        ]

        outputs = {"applier": {"name": applier}}

        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, desc=['n', 'c', "h", "w"])
        result.appendInput(lcs_coeffs, type=TensorProto.FLOAT, desc=[1, 1, "h", "w"])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        return super().build_coordinator(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        H, W = 1080, 1920
        VALUE = 0.5

        coeffs_in   = f"{stage}.lcs_coeffs_in"
        coeffs_out  = f"{stage}.lcs_coeffs"
        coeffs_resz = f"{stage}.lcs_coeffs_resized"

        # Initializer: tensor filled with 0.5
        coeffs_data = [VALUE] * (H * W)
        inits = [
            oh.make_tensor(coeffs_in, TensorProto.FLOAT, [1, 1, 'H', 'W'], coeffs_data),
        ]

        # Identity node to expose original coeffs
        id_node = oh.make_node("Identity", inputs=[coeffs_in], outputs=[coeffs_out], name=f"{stage}_coeffs_id")

        # Scales constant for Resize: shrink both dims by half
        scales_name = f"{stage}.scales"
        scales_init = oh.make_tensor(scales_name, TensorProto.FLOAT, [4], [1, 1, 0.5, 0.5])
        inits.append(scales_init)

        # Explicit empty ROI
        roi = f"{stage}.roi_empty"
        inits += [oh.make_tensor(roi, TensorProto.FLOAT, [0], [])]

        # Resize node
        resize_node = oh.make_node(
            "Resize",
            inputs=[coeffs_in, roi, scales_name],  # roi left empty
            outputs=[coeffs_resz],
            name=f"{stage}_resize_lcs",
            mode="linear",
        )

        nodes = [id_node, resize_node]

        vis = [
            oh.make_tensor_value_info(coeffs_in,   TensorProto.FLOAT, [1,1,'H', 'W']),
            oh.make_tensor_value_info(coeffs_out,  TensorProto.FLOAT, [1,1,'H', 'W']),
            oh.make_tensor_value_info(coeffs_resz, TensorProto.FLOAT, [1,1, 'H // 2', 'W // 2']),
        ]

        outputs = {
            "lcs_coeffs":         {"name": coeffs_out},
            "lcs_coeffs_resized": {"name": coeffs_resz},
        }

        return BuildResult(outputs, nodes, inits, vis)


class LensLCSV2(MicroblockBase):
    """
    LensLCSV2 (v2)
    --------------
    Radial lens shading correction with analytic coefficient generation.

    Needs:
        - applier [n,3,h,w] : image tensor from upstream
        - lcs_params []     : [cx, cy, strength, falloff]

    Provides:
        - applier [n,3,h,w] : corrected image tensor
        - lcs_coeffs [h,w]  : generated coefficient map

    Behavior:
        - build_algo: generates lcs_coeffs from radial parameters
        - build_applier: multiplies applier × lcs_coeffs
    """
    name = 'lens_lcs_v2'
    family = 'lcs_lcs_v2'
    version = 'v2'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Generate lcs_coeffs from radial shading parameters.
        coeff(r) = 1 + strength * (1 - (r / rmax)^falloff)
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        lcs_params = f'{stage}.lcs_params'
        
        # Split parameters
        cx, cy, strength, falloff = [f'{stage}.{p}' for p in ('cx', 'cy', 'strength', 'falloff')]
        nodes.append(oh.make_node('Split', inputs=[lcs_params], outputs=[cx, cy, strength, falloff], 
                                  name=f'{stage}.split_params', axis=0))
        
        # Create coordinate grids
        h_coord = f'{stage}.h_coord'
        w_coord = f'{stage}.w_coord'
        vis.append(oh.make_tensor_value_info(h_coord, TensorProto.FLOAT, ['h']))
        vis.append(oh.make_tensor_value_info(w_coord, TensorProto.FLOAT, ['w']))
        
        # Normalize coordinates to [-1, 1]
        h_norm = f'{stage}.h_norm'
        w_norm = f'{stage}.w_norm'
        h_half = f'{stage}.h_half'
        w_half = f'{stage}.w_half'
        inits.append(oh.make_tensor(h_half, TensorProto.FLOAT, [], [0.5]))
        inits.append(oh.make_tensor(w_half, TensorProto.FLOAT, [], [0.5]))
        
        nodes.append(oh.make_node('Mul', inputs=[h_coord, h_half], outputs=[h_norm], 
                                  name=f'{stage}.mul_h_norm'))
        nodes.append(oh.make_node('Mul', inputs=[w_coord, w_half], outputs=[w_norm], 
                                  name=f'{stage}.mul_w_norm'))
        
        # Create meshgrid (simplified - use outer operations)
        # For simplicity, we'll use a constant center at (0, 0)
        dx = f'{stage}.dx'
        dy = f'{stage}.dy'
        zero = f'{stage}.zero'
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
        
        nodes.append(oh.make_node('Sub', inputs=[h_norm, zero], outputs=[dy], 
                                  name=f'{stage}.sub_y'))
        nodes.append(oh.make_node('Sub', inputs=[w_norm, zero], outputs=[dx], 
                                  name=f'{stage}.sub_x'))
        
        # Calculate radius
        dx2 = f'{stage}.dx2'
        dy2 = f'{stage}.dy2'
        rsq = f'{stage}.rsq'
        r = f'{stage}.r'
        two = f'{stage}.two'
        inits.append(oh.make_tensor(two, TensorProto.FLOAT, [], [2.0]))
        
        nodes.append(oh.make_node('Pow', inputs=[dx, two], outputs=[dx2], 
                                  name=f'{stage}.pow_dx'))
        nodes.append(oh.make_node('Pow', inputs=[dy, two], outputs=[dy2], 
                                  name=f'{stage}.pow_dy'))
        nodes.append(oh.make_node('Add', inputs=[dx2, dy2], outputs=[rsq], 
                                  name=f'{stage}.add_rsq'))
        nodes.append(oh.make_node('Sqrt', inputs=[rsq], outputs=[r], 
                                  name=f'{stage}.sqrt_r'))
        
        # Normalize radius
        rmax = f'{stage}.rmax'
        inits.append(oh.make_tensor(rmax, TensorProto.FLOAT, [], [1.0]))
        rnorm = f'{stage}.rnorm'
        nodes.append(oh.make_node('Div', inputs=[r, rmax], outputs=[rnorm], 
                                  name=f'{stage}.div_rnorm'))
        
        # Calculate correction coefficient
        rpow = f'{stage}.rpow'
        one_minus = f'{stage}.one_minus'
        coeff = f'{stage}.coeff'
        one = f'{stage}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        
        nodes.append(oh.make_node('Pow', inputs=[rnorm, falloff], outputs=[rpow], 
                                  name=f'{stage}.pow_r'))
        nodes.append(oh.make_node('Sub', inputs=[one, rpow], outputs=[one_minus], 
                                  name=f'{stage}.sub_one'))
        nodes.append(oh.make_node('Mul', inputs=[strength, one_minus], outputs=[f'{stage}.mul_strength'], 
                                  name=f'{stage}.mul_strength'))
        nodes.append(oh.make_node('Add', inputs=[one, f'{stage}.mul_strength'], outputs=[coeff], 
                                  name=f'{stage}.add_coeff'))
        
        vis.append(oh.make_tensor_value_info(coeff, TensorProto.FLOAT, ['h', 'w']))
        
        outputs = {
            'lcs_coeffs': {'name': coeff}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(lcs_params, type=TensorProto.FLOAT, shape=[4])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply generated lcs_coeffs to applier (image).
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        lcs_coeffs = f'{upstream}.lcs_coeffs'
        applier = f'{stage}.applier'
        
        nodes.append(oh.make_node('Mul', inputs=[input_image, lcs_coeffs], outputs=[applier], 
                                  name=f'{stage}.mul_apply'))
        
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        
        outputs = {'applier': {'name': applier}}
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(lcs_coeffs, type=TensorProto.FLOAT, shape=['h', 'w'])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        return super().build_coordinator(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)