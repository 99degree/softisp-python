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
    Displacement map generation from radial parameters.

    Needs:
        - applier [n,3,h,w] : image tensor from upstream
        - lsc_params [6]    : [cx, cy, k1, k2, k3, vcm] radial parameters

    Provides:
        - applier [n,3,h,w] : corrected image tensor
        - displacement_map [h,w,2] : generated displacement vectors
        - lsc_gain [h,w] : generated gain factors

    Behavior:
        - build_algo: generates displacement_map and lsc_gain from radial parameters
        - build_applier: applies displacement-based correction

    Displacement Model:
        r^2 = (x-cx)^2 + (y-cy)^2
        scale = 1 + k1*r^2 + k2*r^4 + k3*r^6 + vcm
        dx = (x-cx) * (scale - 1)
        dy = (y-cy) * (scale - 1)
        gain = scale
    """
    name = 'lens_lcs_displacement_v1'
    family = 'lens_lcs_displacement_v1'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Generate displacement_map and lsc_gain from radial parameters.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        lsc_params = f'{stage}.lsc_params'
        
        # Split parameters: cx, cy, k1, k2, k3, vcm
        cx, cy, k1, k2, k3, vcm = [f'{stage}.{p}' for p in ('cx', 'cy', 'k1', 'k2', 'k3', 'vcm')]
        nodes.append(oh.make_node('Split', inputs=[lsc_params], outputs=[cx, cy, k1, k2, k3, vcm],
                                  name=f'{stage}.split_params', axis=0))
        
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
        
        # Calculate displacement from center
        dx_center = f'{stage}.dx_center'
        dy_center = f'{stage}.dy_center'
        nodes.append(oh.make_node('Sub', inputs=[w_norm, cx], outputs=[dx_center],
                                  name=f'{stage}.sub_dx'))
        nodes.append(oh.make_node('Sub', inputs=[h_norm, cy], outputs=[dy_center],
                                  name=f'{stage}.sub_dy'))
        
        # Calculate radius squared
        dx2 = f'{stage}.dx2'
        dy2 = f'{stage}.dy2'
        r2 = f'{stage}.r2'
        two = f'{stage}.two'
        inits.append(oh.make_tensor(two, TensorProto.FLOAT, [], [2.0]))
        
        nodes.append(oh.make_node('Pow', inputs=[dx_center, two], outputs=[dx2],
                                  name=f'{stage}.pow_dx'))
        nodes.append(oh.make_node('Pow', inputs=[dy_center, two], outputs=[dy2],
                                  name=f'{stage}.pow_dy'))
        nodes.append(oh.make_node('Add', inputs=[dx2, dy2], outputs=[r2],
                                  name=f'{stage}.add_r2'))
        
        # Calculate polynomial terms
        k1r2 = f'{stage}.k1r2'
        k2r4 = f'{stage}.k2r4'
        k3r6 = f'{stage}.k3r6'
        four = f'{stage}.four'
        six = f'{stage}.six'
        inits.append(oh.make_tensor(four, TensorProto.FLOAT, [], [4.0]))
        inits.append(oh.make_tensor(six, TensorProto.FLOAT, [], [6.0]))
        
        nodes.append(oh.make_node('Mul', inputs=[k1, r2], outputs=[k1r2],
                                  name=f'{stage}.mul_k1r2'))
        nodes.append(oh.make_node('Pow', inputs=[r2, two], outputs=[f'{stage}.r4'],
                                  name=f'{stage}.pow_r4'))
        nodes.append(oh.make_node('Mul', inputs=[k2, f'{stage}.r4'], outputs=[k2r4],
                                  name=f'{stage}.mul_k2r4'))
        nodes.append(oh.make_node('Pow', inputs=[r2, three], outputs=[f'{stage}.r6'],
                                  name=f'{stage}.pow_r6'))
        nodes.append(oh.make_node('Mul', inputs=[k3, f'{stage}.r6'], outputs=[k3r6],
                                  name=f'{stage}.mul_k3r6'))
        
        # Calculate scale
        poly_term = f'{stage}.poly_term'
        scale = f'{stage}.scale'
        nodes.append(oh.make_node('Add', inputs=[k1r2, k2r4], outputs=[f'{stage}.k1k2'],
                                  name=f'{stage}.add_k1k2'))
        nodes.append(oh.make_node('Add', inputs=[f'{stage}.k1k2', k3r6], outputs=[poly_term],
                                  name=f'{stage}.add_poly'))
        nodes.append(oh.make_node('Add', inputs=[poly_term, vcm], outputs=[scale],
                                  name=f'{stage}.add_scale'))
        
        # Calculate displacement vectors
        one = f'{stage}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        
        scale_minus_one = f'{stage}.scale_minus_one'
        nodes.append(oh.make_node('Sub', inputs=[scale, one], outputs=[scale_minus_one],
                                  name=f'{stage}.sub_scale'))
        
        dx = f'{stage}.dx'
        dy = f'{stage}.dy'
        nodes.append(oh.make_node('Mul', inputs=[dx_center, scale_minus_one], outputs=[dx],
                                  name=f'{stage}.mul_dx'))
        nodes.append(oh.make_node('Mul', inputs=[dy_center, scale_minus_one], outputs=[dy],
                                  name=f'{stage}.mul_dy'))
        
        # Stack into displacement map [h,w,2]
        displacement_map = f'{stage}.displacement_map'
        nodes.append(oh.make_node('Concat', inputs=[dx, dy], outputs=[displacement_map],
                                  name=f'{stage}.concat_disp', axis=-1))
        
        # LSC gain is the scale factor
        lsc_gain = f'{stage}.lsc_gain'
        nodes.append(oh.make_node('Identity', inputs=[scale], outputs=[lsc_gain],
                                  name=f'{stage}.identity_gain'))
        
        vis.append(oh.make_tensor_value_info(displacement_map, TensorProto.FLOAT, ['h', 'w', 2]))
        vis.append(oh.make_tensor_value_info(lsc_gain, TensorProto.FLOAT, ['h', 'w']))
        
        outputs = {
            'displacement_map': {'name': displacement_map},
            'lsc_gain': {'name': lsc_gain}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(lsc_params, type=TensorProto.FLOAT, shape=[6])
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
    Approximated displacement map generation for efficiency.

    Needs:
        - applier [n,3,h,w] : image tensor from upstream
        - lsc_params [4]    : [k1, k2, vcm, strength] simplified parameters

    Provides:
        - applier [n,3,h,w] : corrected image tensor
        - displacement_map [h,w,2] : generated displacement vectors
        - lsc_gain [h,w] : generated gain factors

    Behavior:
        - build_algo: generates approximated displacement_map and lsc_gain
        - build_applier: applies displacement-based correction

    Approximated Model:
        r^2 = x^2 + y^2
        scale = 1 + k1*r^2 + k2*r^4 + vcm
        dx = x * strength * (scale - 1)
        dy = y * strength * (scale - 1)
        gain = scale
    """
    name = 'lens_lcs_displacement_v2'
    family = 'lens_lcs_displacement_v2'
    version = 'v2'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Generate approximated displacement_map and lsc_gain from simplified parameters.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        lsc_params = f'{stage}.lsc_params'
        
        # Split parameters: k1, k2, vcm, strength
        k1, k2, vcm, strength = [f'{stage}.{p}' for p in ('k1', 'k2', 'vcm', 'strength')]
        nodes.append(oh.make_node('Split', inputs=[lsc_params], outputs=[k1, k2, vcm, strength],
                                  name=f'{stage}.split_params', axis=0))
        
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
        
        # Calculate polynomial terms
        k1r2 = f'{stage}.k1r2'
        k2r4 = f'{stage}.k2r4'
        nodes.append(oh.make_node('Mul', inputs=[k1, r2], outputs=[k1r2],
                                  name=f'{stage}.mul_k1r2'))
        nodes.append(oh.make_node('Pow', inputs=[r2, two], outputs=[f'{stage}.r4'],
                                  name=f'{stage}.pow_r4'))
        nodes.append(oh.make_node('Mul', inputs=[k2, f'{stage}.r4'], outputs=[k2r4],
                                  name=f'{stage}.mul_k2r4'))
        
        # Calculate scale
        poly_term = f'{stage}.poly_term'
        scale = f'{stage}.scale'
        nodes.append(oh.make_node('Add', inputs=[k1r2, k2r4], outputs=[poly_term],
                                  name=f'{stage}.add_poly'))
        nodes.append(oh.make_node('Add', inputs=[poly_term, vcm], outputs=[scale],
                                  name=f'{stage}.add_scale'))
        
        # Calculate displacement vectors with strength
        one = f'{stage}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        
        scale_minus_one = f'{stage}.scale_minus_one'
        nodes.append(oh.make_node('Sub', inputs=[scale, one], outputs=[scale_minus_one],
                                  name=f'{stage}.sub_scale'))
        
        displacement_factor = f'{stage}.displacement_factor'
        nodes.append(oh.make_node('Mul', inputs=[strength, scale_minus_one],
                                  outputs=[displacement_factor],
                                  name=f'{stage}.mul_strength'))
        
        dx = f'{stage}.dx'
        dy = f'{stage}.dy'
        nodes.append(oh.make_node('Mul', inputs=[w_norm, displacement_factor], outputs=[dx],
                                  name=f'{stage}.mul_dx'))
        nodes.append(oh.make_node('Mul', inputs=[h_norm, displacement_factor], outputs=[dy],
                                  name=f'{stage}.mul_dy'))
        
        # Stack into displacement map [h,w,2]
        displacement_map = f'{stage}.displacement_map'
        nodes.append(oh.make_node('Concat', inputs=[dx, dy], outputs=[displacement_map],
                                  name=f'{stage}.concat_disp', axis=-1))
        
        # LSC gain is the scale factor
        lsc_gain = f'{stage}.lsc_gain'
        nodes.append(oh.make_node('Identity', inputs=[scale], outputs=[lsc_gain],
                                  name=f'{stage}.identity_gain'))
        
        vis.append(oh.make_tensor_value_info(displacement_map, TensorProto.FLOAT, ['h', 'w', 2]))
        vis.append(oh.make_tensor_value_info(lsc_gain, TensorProto.FLOAT, ['h', 'w']))
        
        outputs = {
            'displacement_map': {'name': displacement_map},
            'lsc_gain': {'name': lsc_gain}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(lsc_params, type=TensorProto.FLOAT, shape=[4])
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