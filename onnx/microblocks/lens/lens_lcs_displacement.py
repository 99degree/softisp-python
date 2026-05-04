from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class LensLCSDisplacementBase(MicroblockBase):
    """
    LensLCSDisplacementBase (v0)
    -----------------------------
    Canonical base microblock for Lens Shading Correction using displacement maps.

    Needs:
        - applier [n,3,h,w] : image tensor from upstream
        - displacement_map [h,w,2] : per-pixel displacement vectors (dx, dy)
        - lsc_gain [h,w] : per-pixel gain factors

    Provides:
        - applier [n,3,h,w] : corrected image tensor

    Behavior:
        - build_algo: declares displacement_map and lsc_gain as external needs
        - build_applier: applies displacement-based correction

    VCM Position Context:
        VCM (Voice Coil Motor) position determines lens focal length and affects
        lens characteristics including vignetting and geometric distortion.
        VCM position changes from min to max during autofocus operation.
    """
    name = 'lens_lcs_displacement_base'
    family = 'lens_lcs_displacement_base'
    version = 'v0'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Base version does not generate displacement_map or lsc_gain.
        Assumes they are provided externally (from coordinator or calibration).
        """
        vis, nodes, inits = ([], [], [])
        disp_map = f'{stage}.displacement_map'
        lsc_gain = f'{stage}.lsc_gain'
        
        vis.append(oh.make_tensor_value_info(disp_map, TensorProto.FLOAT, ['h', 'w', 2]))
        vis.append(oh.make_tensor_value_info(lsc_gain, TensorProto.FLOAT, ['h', 'w']))
        
        outputs = {
            'displacement_map': {'name': disp_map},
            'lsc_gain': {'name': lsc_gain}
        }
        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{prev_stages[0]}.applier')

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply displacement-based lens correction.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        displacement_map = f'{upstream}.displacement_map'
        lsc_gain = f'{upstream}.lsc_gain'
        applier = f'{stage}.applier'
        
        # Split displacement map into dx and dy
        dx = f'{stage}.dx'
        dy = f'{stage}.dy'
        nodes.append(oh.make_node('Split', inputs=[displacement_map], outputs=[dx, dy],
                                  name=f'{stage}.split_disp', axis=-1))
        
        # Create sampling grid from displacement
        # Grid coordinates: x + dx, y + dy
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
        
        # Apply displacement
        x_sample = f'{stage}.x_sample'
        y_sample = f'{stage}.y_sample'
        nodes.append(oh.make_node('Add', inputs=[w_norm, dx], outputs=[x_sample],
                                  name=f'{stage}.add_x_sample'))
        nodes.append(oh.make_node('Add', inputs=[h_norm, dy], outputs=[y_sample],
                                  name=f'{stage}.add_y_sample'))
        
        # Stack into grid [h,w,2]
        grid = f'{stage}.grid'
        nodes.append(oh.make_node('Concat', inputs=[x_sample, y_sample], outputs=[grid],
                                  name=f'{stage}.concat_grid', axis=-1))
        
        # Unsqueeze for GridSample [1,h,w,2]
        grid_expanded = f'{stage}.grid_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[grid], outputs=[grid_expanded],
                                  name=f'{stage}.unsqueeze_grid', axes=[0]))
        
        # Apply GridSample for geometric correction
        geometric_corrected = f'{stage}.geometric_corrected'
        nodes.append(oh.make_node('GridSample', inputs=[input_image, grid_expanded],
                                  outputs=[geometric_corrected],
                                  name=f'{stage}.gridsample', mode='bilinear',
                                  padding_mode='zeros', align_corners=1))
        
        # Apply gain correction
        nodes.append(oh.make_node('Mul', inputs=[geometric_corrected, lsc_gain],
                                  outputs=[applier], name=f'{stage}.mul_gain'))
        
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        outputs = {'applier': {'name': applier}}
        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{prev_stages[0]}.applier')

    def build_coordinator(self, stage: str, prev_stages=None):
        return super().build_coordinator(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)


class LensLCSDisplacementV1(MicroblockBase):
    """
    LensLCSDisplacementV1 (v1)
    ---------------------------
    Displacement map generation from standard LCS coefficients and VCM position.

    Needs:
        - applier [n,3,h,w] : image tensor from upstream
        - lsc_coeffs [2]    : [k1, k2] standard LCS coefficients
        - vcm_pos [1]      : VCM position (lens focal length parameter)

    Provides:
        - applier [n,3,h,w] : corrected image tensor
        - displacement_map [h,w,2] : generated displacement vectors
        - lsc_gain [h,w] : generated gain factors

    Behavior:
        - build_algo: generates displacement_map and lsc_gain from LCS params and VCM
        - build_applier: applies displacement-based correction

    Standard LCS Polynomial Model:
        r^2 = x^2 + y^2
        poly_term = k1*r^2 + k2*r^4
        scale = poly_term + vcm_pos
        dx = x * scale
        dy = y * scale
        gain = scale^2

    VCM Position Context:
        VCM (Voice Coil Motor) position determines lens focal length and affects
        lens characteristics including vignetting and geometric distortion.
        VCM position changes from min to max during autofocus operation.

        The VCM position is used as a base offset in the polynomial to account for
        lens position-dependent characteristics. Different VCM positions may have
        different LCS coefficients (k1, k2) to model position-dependent vignetting.
    """
    name = 'lens_lcs_displacement_v1'
    family = 'lens_lcs_displacement_v1'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Generate displacement_map and lsc_gain from standard LCS coefficients and VCM position.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        lsc_coeffs = f'{stage}.lsc_coeffs'
        vcm_pos = f'{stage}.vcm_pos'
        
        # Split LCS coefficients: k1, k2
        k1, k2 = [f'{stage}.{p}' for p in ('k1', 'k2')]
        nodes.append(oh.make_node('Split', inputs=[lsc_coeffs], outputs=[k1, k2],
                                  name=f'{stage}.split_coeffs', axis=0))
        
        # Create coordinate grids (normalized to [-1, 1])
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
        
        # Calculate radius squared
        x2 = f'{stage}.x2'
        y2 = f'{stage}.y2'
        r2 = f'{stage}.r2'
        two = f'{stage}.two'
        inits.append(oh.make_tensor(two, TensorProto.FLOAT, [], [2.0]))
        
        nodes.append(oh.make_node('Mul', inputs=[w_norm, w_norm], outputs=[x2],
                                  name=f'{stage}.mul_x2'))
        nodes.append(oh.make_node('Mul', inputs=[h_norm, h_norm], outputs=[y2],
                                  name=f'{stage}.mul_y2'))
        nodes.append(oh.make_node('Add', inputs=[x2, y2], outputs=[r2],
                                  name=f'{stage}.add_r2'))
        
        # Calculate polynomial terms (standard LCS model)
        k1r2 = f'{stage}.k1r2'
        k2r4 = f'{stage}.k2r4'
        nodes.append(oh.make_node('Mul', inputs=[k1, r2], outputs=[k1r2],
                                  name=f'{stage}.mul_k1r2'))
        nodes.append(oh.make_node('Mul', inputs=[k2, r2], outputs=[f'{stage}.k2r2'],
                                  name=f'{stage}.mul_k2r2'))
        nodes.append(oh.make_node('Mul', inputs=[f'{stage}.k2r2', r2], outputs=[k2r4],
                                  name=f'{stage}.mul_k2r4'))
        
        # Calculate scale (standard LCS polynomial with VCM position)
        poly_term = f'{stage}.poly_term'
        scale = f'{stage}.scale'
        nodes.append(oh.make_node('Add', inputs=[k1r2, k2r4], outputs=[poly_term],
                                  name=f'{stage}.add_poly'))
        nodes.append(oh.make_node('Add', inputs=[poly_term, vcm_pos], outputs=[scale],
                                  name=f'{stage}.add_scale'))
        
        # Calculate displacement vectors (direct scaling)
        dx = f'{stage}.dx'
        dy = f'{stage}.dy'
        nodes.append(oh.make_node('Mul', inputs=[w_norm, scale], outputs=[dx],
                                  name=f'{stage}.mul_dx'))
        nodes.append(oh.make_node('Mul', inputs=[h_norm, scale], outputs=[dy],
                                  name=f'{stage}.mul_dy'))
        
        # Stack into displacement map [h,w,2]
        displacement_map = f'{stage}.displacement_map'
        nodes.append(oh.make_node('Concat', inputs=[dx, dy], outputs=[displacement_map],
                                  name=f'{stage}.concat_disp', axis=-1))
        
        # LSC gain is scale squared (as per lsc_test.py)
        lsc_gain = f'{stage}.lsc_gain'
        nodes.append(oh.make_node('Mul', inputs=[scale, scale], outputs=[lsc_gain],
                                  name=f'{stage}.mul_gain'))
        
        vis.append(oh.make_tensor_value_info(displacement_map, TensorProto.FLOAT, ['h', 'w', 2]))
        vis.append(oh.make_tensor_value_info(lsc_gain, TensorProto.FLOAT, ['h', 'w']))
        
        outputs = {
            'displacement_map': {'name': displacement_map},
            'lsc_gain': {'name': lsc_gain}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(lsc_coeffs, type=TensorProto.FLOAT, shape=[2])
        result.appendInput(vcm_pos, type=TensorProto.FLOAT, shape=[1])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply displacement-based lens correction.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        displacement_map = f'{upstream}.displacement_map'
        lsc_gain = f'{upstream}.lsc_gain'
        applier = f'{stage}.applier'
        
        # Split displacement map into dx and dy
        dx = f'{stage}.dx'
        dy = f'{stage}.dy'
        nodes.append(oh.make_node('Split', inputs=[displacement_map], outputs=[dx, dy],
                                  name=f'{stage}.split_disp', axis=-1))
        
        # Create sampling grid from displacement
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
        
        # Apply displacement
        x_sample = f'{stage}.x_sample'
        y_sample = f'{stage}.y_sample'
        nodes.append(oh.make_node('Add', inputs=[w_norm, dx], outputs=[x_sample],
                                  name=f'{stage}.add_x_sample'))
        nodes.append(oh.make_node('Add', inputs=[h_norm, dy], outputs=[y_sample],
                                  name=f'{stage}.add_y_sample'))
        
        # Stack into grid [h,w,2]
        grid = f'{stage}.grid'
        nodes.append(oh.make_node('Concat', inputs=[x_sample, y_sample], outputs=[grid],
                                  name=f'{stage}.concat_grid', axis=-1))
        
        # Unsqueeze for GridSample [1,h,w,2]
        grid_expanded = f'{stage}.grid_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[grid], outputs=[grid_expanded],
                                  name=f'{stage}.unsqueeze_grid', axes=[0]))
        
        # Apply GridSample for geometric correction
        geometric_corrected = f'{stage}.geometric_corrected'
        nodes.append(oh.make_node('GridSample', inputs=[input_image, grid_expanded],
                                  outputs=[geometric_corrected],
                                  name=f'{stage}.gridsample', mode='bilinear',
                                  padding_mode='zeros', align_corners=1))
        
        # Apply gain correction
        nodes.append(oh.make_node('Mul', inputs=[geometric_corrected, lsc_gain],
                                  outputs=[applier], name=f'{stage}.mul_gain'))
        
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        outputs = {'applier': {'name': applier}}
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(displacement_map, type=TensorProto.FLOAT, shape=['h', 'w', 2])
        result.appendInput(lsc_gain, type=TensorProto.FLOAT, shape=['h', 'w'])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        return super().build_coordinator(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)


class LensLCSDisplacementV2(MicroblockBase):
    """
    LensLCSDisplacementV2 (v2)
    ---------------------------
    Extended displacement map generation with additional LCS coefficients and VCM position.

    Needs:
        - applier [n,3,h,w] : image tensor from upstream
        - lsc_coeffs [3]    : [k1, k2, k3] extended LCS coefficients
        - vcm_pos [1]      : VCM position (lens focal length parameter)

    Provides:
        - applier [n,3,h,w] : corrected image tensor
        - displacement_map [h,w,2] : generated displacement vectors
        - lsc_gain [h,w] : generated gain factors

    Behavior:
        - build_algo: generates displacement_map and lsc_gain from extended LCS params and VCM
        - build_applier: applies displacement-based correction

    Extended LCS Polynomial Model:
        r^2 = x^2 + y^2
        poly_term = k1*r^2 + k2*r^4 + k3*r^6
        scale = poly_term + vcm_pos
        dx = x * scale
        dy = y * scale
        gain = scale^2

    VCM Position Context:
        VCM (Voice Coil Motor) position determines lens focal length and affects
        lens characteristics including vignetting and geometric distortion.
        VCM position changes from min to max during autofocus operation.

        The VCM position is used as a base offset in the polynomial to account for
        lens position-dependent characteristics. Different VCM positions may have
        different LCS coefficients (k1, k2, k3) to model position-dependent vignetting.

        Extended model with k3 provides higher-order correction for more complex
        lens characteristics at different VCM positions.
    """
    name = 'lens_lcs_displacement_v2'
    family = 'lens_lcs_displacement_v2'
    version = 'v2'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Generate displacement_map and lsc_gain from extended LCS coefficients and VCM position.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        lsc_coeffs = f'{stage}.lsc_coeffs'
        vcm_pos = f'{stage}.vcm_pos'
        
        # Split LCS coefficients: k1, k2, k3
        k1, k2, k3 = [f'{stage}.{p}' for p in ('k1', 'k2', 'k3')]
        nodes.append(oh.make_node('Split', inputs=[lsc_coeffs], outputs=[k1, k2, k3],
                                  name=f'{stage}.split_coeffs', axis=0))
        
        # Create coordinate grids (normalized to [-1, 1])
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
        
        # Calculate radius squared
        x2 = f'{stage}.x2'
        y2 = f'{stage}.y2'
        r2 = f'{stage}.r2'
        two = f'{stage}.two'
        inits.append(oh.make_tensor(two, TensorProto.FLOAT, [], [2.0]))
        
        nodes.append(oh.make_node('Mul', inputs=[w_norm, w_norm], outputs=[x2],
                                  name=f'{stage}.mul_x2'))
        nodes.append(oh.make_node('Mul', inputs=[h_norm, h_norm], outputs=[y2],
                                  name=f'{stage}.mul_y2'))
        nodes.append(oh.make_node('Add', inputs=[x2, y2], outputs=[r2],
                                  name=f'{stage}.add_r2'))
        
        # Calculate polynomial terms (extended LCS model)
        k1r2 = f'{stage}.k1r2'
        k2r4 = f'{stage}.k2r4'
        k3r6 = f'{stage}.k3r6'
        three = f'{stage}.three'
        inits.append(oh.make_tensor(three, TensorProto.FLOAT, [], [3.0]))
        
        nodes.append(oh.make_node('Mul', inputs=[k1, r2], outputs=[k1r2],
                                  name=f'{stage}.mul_k1r2'))
        nodes.append(oh.make_node('Mul', inputs=[k2, r2], outputs=[f'{stage}.k2r2'],
                                  name=f'{stage}.mul_k2r2'))
        nodes.append(oh.make_node('Mul', inputs=[f'{stage}.k2r2', r2], outputs=[k2r4],
                                  name=f'{stage}.mul_k2r4'))
        nodes.append(oh.make_node('Mul', inputs=[k3, r2], outputs=[f'{stage}.k3r2'],
                                  name=f'{stage}.mul_k3r2'))
        nodes.append(oh.make_node('Mul', inputs=[f'{stage}.k3r2', r2], outputs=[f'{stage}.k3r4'],
                                  name=f'{stage}.mul_k3r4'))
        nodes.append(oh.make_node('Mul', inputs=[f'{stage}.k3r4', r2], outputs=[k3r6],
                                  name=f'{stage}.mul_k3r6'))
        
        # Calculate scale (extended LCS polynomial with VCM position)
        poly_term = f'{stage}.poly_term'
        scale = f'{stage}.scale'
        nodes.append(oh.make_node('Add', inputs=[k1r2, k2r4], outputs=[f'{stage}.k1k2'],
                                  name=f'{stage}.add_k1k2'))
        nodes.append(oh.make_node('Add', inputs=[f'{stage}.k1k2', k3r6], outputs=[poly_term],
                                  name=f'{stage}.add_poly'))
        nodes.append(oh.make_node('Add', inputs=[poly_term, vcm_pos], outputs=[scale],
                                  name=f'{stage}.add_scale'))
        
        # Calculate displacement vectors (direct scaling)
        dx = f'{stage}.dx'
        dy = f'{stage}.dy'
        nodes.append(oh.make_node('Mul', inputs=[w_norm, scale], outputs=[dx],
                                  name=f'{stage}.mul_dx'))
        nodes.append(oh.make_node('Mul', inputs=[h_norm, scale], outputs=[dy],
                                  name=f'{stage}.mul_dy'))
        
        # Stack into displacement map [h,w,2]
        displacement_map = f'{stage}.displacement_map'
        nodes.append(oh.make_node('Concat', inputs=[dx, dy], outputs=[displacement_map],
                                  name=f'{stage}.concat_disp', axis=-1))
        
        # LSC gain is scale squared (as per lsc_test.py)
        lsc_gain = f'{stage}.lsc_gain'
        nodes.append(oh.make_node('Mul', inputs=[scale, scale], outputs=[lsc_gain],
                                  name=f'{stage}.mul_gain'))
        
        vis.append(oh.make_tensor_value_info(displacement_map, TensorProto.FLOAT, ['h', 'w', 2]))
        vis.append(oh.make_tensor_value_info(lsc_gain, TensorProto.FLOAT, ['h', 'w']))
        
        outputs = {
            'displacement_map': {'name': displacement_map},
            'lsc_gain': {'name': lsc_gain}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(lsc_coeffs, type=TensorProto.FLOAT, shape=[3])
        result.appendInput(vcm_pos, type=TensorProto.FLOAT, shape=[1])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply displacement-based lens correction.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        displacement_map = f'{upstream}.displacement_map'
        lsc_gain = f'{upstream}.lsc_gain'
        applier = f'{stage}.applier'
        
        # Split displacement map into dx and dy
        dx = f'{stage}.dx'
        dy = f'{stage}.dy'
        nodes.append(oh.make_node('Split', inputs=[displacement_map], outputs=[dx, dy],
                                  name=f'{stage}.split_disp', axis=-1))
        
        # Create sampling grid from displacement
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
        
        # Apply displacement
        x_sample = f'{stage}.x_sample'
        y_sample = f'{stage}.y_sample'
        nodes.append(oh.make_node('Add', inputs=[w_norm, dx], outputs=[x_sample],
                                  name=f'{stage}.add_x_sample'))
        nodes.append(oh.make_node('Add', inputs=[h_norm, dy], outputs=[y_sample],
                                  name=f'{stage}.add_y_sample'))
        
        # Stack into grid [h,w,2]
        grid = f'{stage}.grid'
        nodes.append(oh.make_node('Concat', inputs=[x_sample, y_sample], outputs=[grid],
                                  name=f'{stage}.concat_grid', axis=-1))
        
        # Unsqueeze for GridSample [1,h,w,2]
        grid_expanded = f'{stage}.grid_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[grid], outputs=[grid_expanded],
                                  name=f'{stage}.unsqueeze_grid', axes=[0]))
        
        # Apply GridSample for geometric correction
        geometric_corrected = f'{stage}.geometric_corrected'
        nodes.append(oh.make_node('GridSample', inputs=[input_image, grid_expanded],
                                  outputs=[geometric_corrected],
                                  name=f'{stage}.gridsample', mode='bilinear',
                                  padding_mode='zeros', align_corners=1))
        
        # Apply gain correction
        nodes.append(oh.make_node('Mul', inputs=[geometric_corrected, lsc_gain],
                                  outputs=[applier], name=f'{stage}.mul_gain'))
        
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        outputs = {'applier': {'name': applier}}
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(displacement_map, type=TensorProto.FLOAT, shape=['h', 'w', 2])
        result.appendInput(lsc_gain, type=TensorProto.FLOAT, shape=['h', 'w'])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        return super().build_coordinator(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)