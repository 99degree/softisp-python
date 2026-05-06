from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class LensGDCBase(MicroblockBase):
    """
    LensGDCBase (v0)
    ----------------
    Canonical base microblock for Geometric Distortion Correction (GDC).

    Needs:
        - applier [n,3,h,w] : image tensor from upstream
        - gdc_grid [h,w,2]  : geometric distortion correction grid

    Provides:
        - applier [n,3,h,w] : geometrically corrected image tensor

    Behavior:
        - build_algo: declares gdc_grid as an external need (no generation here)
        - build_applier: applies geometric correction using GridSample
    """
    name = 'lens_gdc_base'
    family = 'lens_gdc_base'
    version = 'v0'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Base version does not generate gdc_grid.
        Assumes gdc_grid is provided externally (from coordinator or calibration).
        """
        vis, nodes, inits = ([], [], [])
        grid_name = f'{stage}.gdc_grid'
        vis.append(oh.make_tensor_value_info(grid_name, TensorProto.FLOAT, ['h', 'w', 2]))
        outputs = {'gdc_grid': {'name': grid_name}}
        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{prev_stages[0]}.applier')

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply geometric distortion correction using GridSample.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        gdc_grid = f'{upstream}.gdc_grid'
        applier = f'{stage}.applier'
        
        # Transpose grid from [h,w,2] to [1,h,w,2] for GridSample
        grid_expanded = f'{stage}.grid_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[gdc_grid], outputs=[grid_expanded], 
                                  name=f'{stage}.unsqueeze_grid', axes=[0]))
        
        # Apply GridSample for geometric transformation
        nodes.append(oh.make_node('GridSample', inputs=[input_image, grid_expanded], outputs=[applier],
                                  name=f'{stage}.gridsample', mode='bilinear', 
                                  padding_mode='zeros', align_corners=1))
        
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        outputs = {'applier': {'name': applier}}
        return BuildResult(outputs, nodes, inits, vis).appendInput(f'{prev_stages[0]}.applier')

    def build_coordinator(self, stage: str, prev_stages=None):
        return super().build_coordinator(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)


class LensGDCV1(MicroblockBase):
    """
    LensGDCV1 (v1)
    --------------
    Radial distortion correction with polynomial model.

    Needs:
        - applier [n,3,h,w] : image tensor from upstream
        - gdc_params [4]    : [k1, k2, k3, vcm] distortion parameters

    Provides:
        - applier [n,3,h,w] : geometrically corrected image tensor
        - gdc_grid [h,w,2]  : generated correction grid

    Behavior:
        - build_algo: generates gdc_grid from radial distortion parameters
        - build_applier: applies geometric correction using GridSample

    Radial distortion model:
        r^2 = x^2 + y^2
        scale = 1 + k1*r^2 + k2*r^4 + k3*r^6 + vcm
        x' = x * scale
        y' = y * scale
    """
    name = 'lens_gdc_v1'
    family = 'lens_gdc_v1'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Generate gdc_grid from radial distortion parameters.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        gdc_params = f'{stage}.gdc_params'
        
        # Split parameters: k1, k2, k3, vcm
        k1, k2, k3, vcm = [f'{stage}.{p}' for p in ('k1', 'k2', 'k3', 'vcm')]
        nodes.append(oh.make_node('Split', inputs=[gdc_params], outputs=[k1, k2, k3, vcm],
                                  name=f'{stage}.split_params', axis=0))
        
        # Create coordinate grids (normalized to [-1, 1])
        # For simplicity, we'll use constant grids
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
        x = f'{stage}.x'
        y = f'{stage}.y'
        x2 = f'{stage}.x2'
        y2 = f'{stage}.y2'
        r2 = f'{stage}.r2'
        two = f'{stage}.two'
        inits.append(oh.make_tensor(two, TensorProto.FLOAT, [], [2.0]))
        
        nodes.append(oh.make_node('Pow', inputs=[w_norm, two], outputs=[x2],
                                  name=f'{stage}.pow_x'))
        nodes.append(oh.make_node('Pow', inputs=[h_norm, two], outputs=[y2],
                                  name=f'{stage}.pow_y'))
        nodes.append(oh.make_node('Add', inputs=[x2, y2], outputs=[r2],
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
        
        # Calculate corrected coordinates
        x_prime = f'{stage}.x_prime'
        y_prime = f'{stage}.y_prime'
        nodes.append(oh.make_node('Mul', inputs=[w_norm, scale], outputs=[x_prime],
                                  name=f'{stage}.mul_x_prime'))
        nodes.append(oh.make_node('Mul', inputs=[h_norm, scale], outputs=[y_prime],
                                  name=f'{stage}.mul_y_prime'))
        
        # Stack into grid [h,w,2]
        gdc_grid = f'{stage}.gdc_grid'
        nodes.append(oh.make_node('Concat', inputs=[x_prime, y_prime], outputs=[gdc_grid],
                                  name=f'{stage}.concat_grid', axis=-1))
        
        vis.append(oh.make_tensor_value_info(gdc_grid, TensorProto.FLOAT, ['h', 'w', 2]))
        
        outputs = {
            'gdc_grid': {'name': gdc_grid}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(gdc_params, type=TensorProto.FLOAT, shape=[4])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply geometric distortion correction using GridSample.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        gdc_grid = f'{upstream}.gdc_grid'
        applier = f'{stage}.applier'
        
        # Transpose grid from [h,w,2] to [1,h,w,2] for GridSample
        grid_expanded = f'{stage}.grid_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[gdc_grid], outputs=[grid_expanded],
                                  name=f'{stage}.unsqueeze_grid', axes=[0]))
        
        # Apply GridSample for geometric transformation
        nodes.append(oh.make_node('GridSample', inputs=[input_image, grid_expanded], outputs=[applier],
                                  name=f'{stage}.gridsample', mode='bilinear',
                                  padding_mode='zeros', align_corners=1))
        
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        outputs = {'applier': {'name': applier}}
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(gdc_grid, type=TensorProto.FLOAT, shape=['h', 'w', 2])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        return super().build_coordinator(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)


class LensGDCV2(MicroblockBase):
    """
    LensGDCV2 (v2)
    --------------
    Fisheye distortion correction with advanced model.

    Needs:
        - applier [n,3,h,w] : image tensor from upstream
        - gdc_params [5]    : [k1, k2, k3, k4, vcm] distortion parameters

    Provides:
        - applier [n,3,h,w] : geometrically corrected image tensor
        - gdc_grid [h,w,2]  : generated correction grid

    Behavior:
        - build_algo: generates gdc_grid from fisheye distortion parameters
        - build_applier: applies geometric correction using GridSample

    Fisheye distortion model:
        r = sqrt(x^2 + y^2)
        theta = atan(r)
        r' = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
        scale = r' / r + vcm
        x' = x * scale
        y' = y * scale
    """
    name = 'lens_gdc_v2'
    family = 'lens_gdc_v2'
    version = 'v2'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Generate gdc_grid from fisheye distortion parameters.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        gdc_params = f'{stage}.gdc_params'
        
        # Split parameters: k1, k2, k3, k4, vcm
        k1, k2, k3, k4, vcm = [f'{stage}.{p}' for p in ('k1', 'k2', 'k3', 'k4', 'vcm')]
        nodes.append(oh.make_node('Split', inputs=[gdc_params], outputs=[k1, k2, k3, k4, vcm],
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
        
        # Calculate radius
        x2 = f'{stage}.x2'
        y2 = f'{stage}.y2'
        r2 = f'{stage}.r2'
        r = f'{stage}.r'
        two = f'{stage}.two'
        inits.append(oh.make_tensor(two, TensorProto.FLOAT, [], [2.0]))
        
        nodes.append(oh.make_node('Mul', inputs=[w_norm, w_norm], outputs=[x2],
                                  name=f'{stage}.mul_x2'))
        nodes.append(oh.make_node('Mul', inputs=[h_norm, h_norm], outputs=[y2],
                                  name=f'{stage}.mul_y2'))
        nodes.append(oh.make_node('Add', inputs=[x2, y2], outputs=[r2],
                                  name=f'{stage}.add_r2'))
        nodes.append(oh.make_node('Sqrt', inputs=[r2], outputs=[r],
                                  name=f'{stage}.sqrt_r'))
        
        # Calculate theta = atan(r)
        theta = f'{stage}.theta'
        nodes.append(oh.make_node('Atan', inputs=[r], outputs=[theta],
                                  name=f'{stage}.atan_theta'))
        
        # Calculate polynomial terms in theta
        theta2 = f'{stage}.theta2'
        theta4 = f'{stage}.theta4'
        theta6 = f'{stage}.theta6'
        theta8 = f'{stage}.theta8'
        
        nodes.append(oh.make_node('Mul', inputs=[theta, theta], outputs=[theta2],
                                  name=f'{stage}.mul_theta2'))
        nodes.append(oh.make_node('Mul', inputs=[theta2, theta2], outputs=[theta4],
                                  name=f'{stage}.mul_theta4'))
        nodes.append(oh.make_node('Mul', inputs=[theta4, theta2], outputs=[theta6],
                                  name=f'{stage}.mul_theta6'))
        nodes.append(oh.make_node('Mul', inputs=[theta6, theta2], outputs=[theta8],
                                  name=f'{stage}.mul_theta8'))
        
        # Calculate polynomial correction
        k1t2 = f'{stage}.k1t2'
        k2t4 = f'{stage}.k2t4'
        k3t6 = f'{stage}.k3t6'
        k4t8 = f'{stage}.k4t8'
        
        nodes.append(oh.make_node('Mul', inputs=[k1, theta2], outputs=[k1t2],
                                  name=f'{stage}.mul_k1t2'))
        nodes.append(oh.make_node('Mul', inputs=[k2, theta4], outputs=[k2t4],
                                  name=f'{stage}.mul_k2t4'))
        nodes.append(oh.make_node('Mul', inputs=[k3, theta6], outputs=[k3t6],
                                  name=f'{stage}.mul_k3t6'))
        nodes.append(oh.make_node('Mul', inputs=[k4, theta8], outputs=[k4t8],
                                  name=f'{stage}.mul_k4t8'))
        
        # Calculate r' = theta * (1 + polynomial)
        poly = f'{stage}.poly'
        one = f'{stage}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        
        nodes.append(oh.make_node('Add', inputs=[one, k1t2], outputs=[f'{stage}.p1'],
                                  name=f'{stage}.add_p1'))
        nodes.append(oh.make_node('Add', inputs=[f'{stage}.p1', k2t4], outputs=[f'{stage}.p2'],
                                  name=f'{stage}.add_p2'))
        nodes.append(oh.make_node('Add', inputs=[f'{stage}.p2', k3t6], outputs=[f'{stage}.p3'],
                                  name=f'{stage}.add_p3'))
        nodes.append(oh.make_node('Add', inputs=[f'{stage}.p3', k4t8], outputs=[poly],
                                  name=f'{stage}.add_poly'))
        
        r_prime = f'{stage}.r_prime'
        nodes.append(oh.make_node('Mul', inputs=[theta, poly], outputs=[r_prime],
                                  name=f'{stage}.mul_r_prime'))
        
        # Calculate scale = r' / r + vcm
        scale = f'{stage}.scale'
        nodes.append(oh.make_node('Div', inputs=[r_prime, r], outputs=[f'{stage}.r_ratio'],
                                  name=f'{stage}.div_r_ratio'))
        nodes.append(oh.make_node('Add', inputs=[f'{stage}.r_ratio', vcm], outputs=[scale],
                                  name=f'{stage}.add_scale'))
        
        # Calculate corrected coordinates
        x_prime = f'{stage}.x_prime'
        y_prime = f'{stage}.y_prime'
        nodes.append(oh.make_node('Mul', inputs=[w_norm, scale], outputs=[x_prime],
                                  name=f'{stage}.mul_x_prime'))
        nodes.append(oh.make_node('Mul', inputs=[h_norm, scale], outputs=[y_prime],
                                  name=f'{stage}.mul_y_prime'))
        
        # Stack into grid [h,w,2]
        gdc_grid = f'{stage}.gdc_grid'
        nodes.append(oh.make_node('Concat', inputs=[x_prime, y_prime], outputs=[gdc_grid],
                                  name=f'{stage}.concat_grid', axis=-1))
        
        vis.append(oh.make_tensor_value_info(gdc_grid, TensorProto.FLOAT, ['h', 'w', 2]))
        
        outputs = {
            'gdc_grid': {'name': gdc_grid}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(gdc_params, type=TensorProto.FLOAT, shape=[5])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply geometric distortion correction using GridSample.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f'{upstream}.applier'
        gdc_grid = f'{upstream}.gdc_grid'
        applier = f'{stage}.applier'
        
        # Transpose grid from [h,w,2] to [1,h,w,2] for GridSample
        grid_expanded = f'{stage}.grid_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[gdc_grid], outputs=[grid_expanded],
                                  name=f'{stage}.unsqueeze_grid', axes=[0]))
        
        # Apply GridSample for geometric transformation
        nodes.append(oh.make_node('GridSample', inputs=[input_image, grid_expanded], outputs=[applier],
                                  name=f'{stage}.gridsample', mode='bilinear',
                                  padding_mode='zeros', align_corners=1))
        
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        outputs = {'applier': {'name': applier}}
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(gdc_grid, type=TensorProto.FLOAT, shape=['h', 'w', 2])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        return super().build_coordinator(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)