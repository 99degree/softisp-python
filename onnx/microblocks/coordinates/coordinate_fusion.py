from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class CoordinateFusionBase(MicroblockBase):
    """
    CoordinateFusionBase (v0)
    --------------------------
    COORDINATES: Fuse GDC and Deshake into single coordinate grid.
    
    Position: Between Algo and Applier, fuses static and dynamic corrections.
    
    Purpose: Merge GDC (lens distortion) and Deshake (motion compensation)
             into a single coordinate grid for one-pass correction.
    
    Needs:
        - gdc_coeffs [4] : GDC coefficients [k1, k2, p1, p2]
        - homography [3,3] : Deshake homography matrix (OpenCV format)
        - image_size [2] : [height, width] for grid generation

    Provides:
        - fused_grid [h,w,2] : Fused coordinate grid for GridSample
        - gdc_grid [h,w,2] : GDC-only coordinate grid (for debugging)
        - deshake_grid [h,w,2] : Deshake-only coordinate grid (for debugging)

    Behavior:
        - build_algo: Not used (algo extracts homography from frames)
        - build_coordinator: Create fused coordinate grid
        - build_applier: Not used (applier handles application)

    Transformation Composition:
        P_final = T_deshake * T_gdc * P_source
        
        Where:
        - T_gdc: GDC distortion transformation (lens correction)
        - T_deshake: Deshake homography transformation (motion compensation)
        - P_source: Source pixel coordinates
        - P_final: Final pixel coordinates for sampling

    Grid Generation Steps:
        1. Create normalized identity grid [-1, 1] x [-1, 1]
        2. Apply GDC distortion to grid points
        3. Apply Deshake homography to distorted points
        4. Output fused grid for GridSample

    Complexity: ~30-40 ONNX nodes
    Use Case: Real-time fused GDC + Deshake correction
    """
    name = 'coordinate_fusion_base'
    family = 'coordinate_fusion_base'
    version = 'v0'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts homography from frames.
        """
        return BuildResult({}, [], [], [])

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Create fused coordinate grid combining GDC and Deshake.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        
        gdc_coeffs = f'{upstream}.gdc_coeffs'
        homography = f'{upstream}.homography'
        image_size = f'{upstream}.image_size'
        
        # Extract image dimensions
        height = f'{stage}.height'
        width = f'{stage}.width'
        nodes.append(oh.make_node('Slice', inputs=[image_size], outputs=[height],
                                  name=f'{stage}.slice_height',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[image_size], outputs=[width],
                                  name=f'{stage}.slice_width',
                                  starts=[1], ends=[2], axes=[0]))
        
        # Create coordinate grid
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
        
        # Stack into coordinate grid [h,w,2]
        identity_grid = f'{stage}.identity_grid'
        nodes.append(oh.make_node('Concat', inputs=[w_norm, h_norm], outputs=[identity_grid],
                                  name=f'{stage}.concat_identity_grid', axis=-1))
        
        # Extract GDC coefficients
        k1 = f'{stage}.k1'
        k2 = f'{stage}.k2'
        p1 = f'{stage}.p1'
        p2 = f'{stage}.p2'
        nodes.append(oh.make_node('Slice', inputs=[gdc_coeffs], outputs=[k1],
                                  name=f'{stage}.slice_k1',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gdc_coeffs], outputs=[k2],
                                  name=f'{stage}.slice_k2',
                                  starts=[1], ends=[2], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gdc_coeffs], outputs=[p1],
                                  name=f'{stage}.slice_p1',
                                  starts=[2], ends=[3], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gdc_coeffs], outputs=[p2],
                                  name=f'{stage}.slice_p2',
                                  starts=[3], ends=[4], axes=[0]))
        
        # Extract x and y from identity grid
        x = f'{stage}.x'
        y = f'{stage}.y'
        nodes.append(oh.make_node('Slice', inputs=[identity_grid], outputs=[x],
                                  name=f'{stage}.slice_x',
                                  starts=[0], ends=[1], axes=[-1]))
        nodes.append(oh.make_node('Slice', inputs=[identity_grid], outputs=[y],
                                  name=f'{stage}.slice_y',
                                  starts=[1], ends=[2], axes=[-1]))
        
        # Calculate radius squared
        x_sq = f'{stage}.x_sq'
        y_sq = f'{stage}.y_sq'
        r_sq = f'{stage}.r_sq'
        nodes.append(oh.make_node('Mul', inputs=[x, x], outputs=[x_sq],
                                  name=f'{stage}.mul_x_sq'))
        nodes.append(oh.make_node('Mul', inputs=[y, y], outputs=[y_sq],
                                  name=f'{stage}.mul_y_sq'))
        nodes.append(oh.make_node('Add', inputs=[x_sq, y_sq], outputs=[r_sq],
                                  name=f'{stage}.add_r_sq'))
        
        # Calculate radial distortion factor
        # r_dist = 1 + k1 * r^2 + k2 * r^4
        r4 = f'{stage}.r4'
        nodes.append(oh.make_node('Mul', inputs=[r_sq, r_sq], outputs=[r4],
                                  name=f'{stage}.mul_r4'))
        
        k1_term = f'{stage}.k1_term'
        k2_term = f'{stage}.k2_term'
        nodes.append(oh.make_node('Mul', inputs=[k1, r_sq], outputs=[k1_term],
                                  name=f'{stage}.mul_k1_term'))
        nodes.append(oh.make_node('Mul', inputs=[k2, r4], outputs=[k2_term],
                                  name=f'{stage}.mul_k2_term'))
        
        radial_factor = f'{stage}.radial_factor'
        one = f'{stage}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[radial_factor],
                                  name=f'{stage}.identity_radial_factor'))
        nodes.append(oh.make_node('Add', inputs=[radial_factor, k1_term], outputs=[radial_factor],
                                  name=f'{stage}.add_k1_term'))
        nodes.append(oh.make_node('Add', inputs=[radial_factor, k2_term], outputs=[radial_factor],
                                  name=f'{stage}.add_k2_term'))
        
        # Calculate tangential distortion
        # x_tan = 2 * p1 * x * y + p2 * (r^2 + 2 * x^2)
        # y_tan = p1 * (r^2 + 2 * y^2) + 2 * p2 * x * y
        xy = f'{stage}.xy'
        nodes.append(oh.make_node('Mul', inputs=[x, y], outputs=[xy],
                                  name=f'{stage}.mul_xy'))
        
        two = f'{stage}.two'
        inits.append(oh.make_tensor(two, TensorProto.FLOAT, [], [2.0]))
        
        two_xy = f'{stage}.two_xy'
        nodes.append(oh.make_node('Mul', inputs=[two, xy], outputs=[two_xy],
                                  name=f'{stage}.mul_two_xy'))
        
        two_x_sq = f'{stage}.two_x_sq'
        nodes.append(oh.make_node('Mul', inputs=[two, x_sq], outputs=[two_x_sq],
                                  name=f'{stage}.mul_two_x_sq'))
        
        two_y_sq = f'{stage}.two_y_sq'
        nodes.append(oh.make_node('Mul', inputs=[two, y_sq], outputs=[two_y_sq],
                                  name=f'{stage}.mul_two_y_sq'))
        
        r2_plus_2x2 = f'{stage}.r2_plus_2x2'
        r2_plus_2y2 = f'{stage}.r2_plus_2y2'
        nodes.append(oh.make_node('Add', inputs=[r_sq, two_x_sq], outputs=[r2_plus_2x2],
                                  name=f'{stage}.add_r2_plus_2x2'))
        nodes.append(oh.make_node('Add', inputs=[r_sq, two_y_sq], outputs=[r2_plus_2y2],
                                  name=f'{stage}.add_r2_plus_2y2'))
        
        x_tan = f'{stage}.x_tan'
        y_tan = f'{stage}.y_tan'
        p1_term = f'{stage}.p1_term'
        p2_term = f'{stage}.p2_term'
        nodes.append(oh.make_node('Mul', inputs=[p1, two_xy], outputs=[p1_term],
                                  name=f'{stage}.mul_p1_term'))
        nodes.append(oh.make_node('Mul', inputs=[p2, r2_plus_2x2], outputs=[p2_term],
                                  name=f'{stage}.mul_p2_term'))
        nodes.append(oh.make_node('Add', inputs=[p1_term, p2_term], outputs=[x_tan],
                                  name=f'{stage}.add_x_tan'))
        
        nodes.append(oh.make_node('Mul', inputs=[p1, r2_plus_2y2], outputs=[p1_term],
                                  name=f'{stage}.mul_p1_term_y'))
        nodes.append(oh.make_node('Mul', inputs=[p2, two_xy], outputs=[p2_term],
                                  name=f'{stage}.mul_p2_term_y'))
        nodes.append(oh.make_node('Add', inputs=[p1_term, p2_term], outputs=[y_tan],
                                  name=f'{stage}.add_y_tan'))
        
        # Apply GDC distortion
        # x_gdc = x * radial_factor + x_tan
        # y_gdc = y * radial_factor + y_tan
        x_radial = f'{stage}.x_radial'
        y_radial = f'{stage}.y_radial'
        nodes.append(oh.make_node('Mul', inputs=[x, radial_factor], outputs=[x_radial],
                                  name=f'{stage}.mul_x_radial'))
        nodes.append(oh.make_node('Mul', inputs=[y, radial_factor], outputs=[y_radial],
                                  name=f'{stage}.mul_y_radial'))
        
        x_gdc = f'{stage}.x_gdc'
        y_gdc = f'{stage}.y_gdc'
        nodes.append(oh.make_node('Add', inputs=[x_radial, x_tan], outputs=[x_gdc],
                                  name=f'{stage}.add_x_gdc'))
        nodes.append(oh.make_node('Add', inputs=[y_radial, y_tan], outputs=[y_gdc],
                                  name=f'{stage}.add_y_gdc'))
        
        # Stack into GDC grid [h,w,2]
        gdc_grid = f'{stage}.gdc_grid'
        nodes.append(oh.make_node('Concat', inputs=[x_gdc, y_gdc], outputs=[gdc_grid],
                                  name=f'{stage}.concat_gdc_grid', axis=-1))
        
        # Extract homography elements
        h00 = f'{stage}.h00'
        h01 = f'{stage}.h01'
        h02 = f'{stage}.h02'
        h10 = f'{stage}.h10'
        h11 = f'{stage}.h11'
        h12 = f'{stage}.h12'
        h20 = f'{stage}.h20'
        h21 = f'{stage}.h21'
        h22 = f'{stage}.h22'
        
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h00],
                                  name=f'{stage}.slice_h00',
                                  starts=[0, 0], ends=[1, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h01],
                                  name=f'{stage}.slice_h01',
                                  starts=[0, 1], ends=[1, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h02],
                                  name=f'{stage}.slice_h02',
                                  starts=[0, 2], ends=[1, 3], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h10],
                                  name=f'{stage}.slice_h10',
                                  starts=[1, 0], ends=[2, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h11],
                                  name=f'{stage}.slice_h11',
                                  starts=[1, 1], ends=[2, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h12],
                                  name=f'{stage}.slice_h12',
                                  starts=[1, 2], ends=[2, 3], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h20],
                                  name=f'{stage}.slice_h20',
                                  starts=[2, 0], ends=[3, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h21],
                                  name=f'{stage}.slice_h21',
                                  starts=[2, 1], ends=[3, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h22],
                                  name=f'{stage}.slice_h22',
                                  starts=[2, 2], ends=[3, 3], axes=[0, 1]))
        
        # Apply Deshake homography to GDC-distorted points
        # [x', y', w']^T = H * [x_gdc, y_gdc, 1]^T
        ones = f'{stage}.ones'
        inits.append(oh.make_tensor(ones, TensorProto.FLOAT, [], [1.0]))
        
        x_term1 = f'{stage}.x_term1'
        x_term2 = f'{stage}.x_term2'
        x_term3 = f'{stage}.x_term3'
        nodes.append(oh.make_node('Mul', inputs=[h00, x_gdc], outputs=[x_term1],
                                  name=f'{stage}.mul_x_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h01, y_gdc], outputs=[x_term2],
                                  name=f'{stage}.mul_x_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h02, ones], outputs=[x_term3],
                                  name=f'{stage}.mul_x_term3'))
        
        x_prime = f'{stage}.x_prime'
        x_sum = f'{stage}.x_sum'
        nodes.append(oh.make_node('Add', inputs=[x_term1, x_term2], outputs=[x_sum],
                                  name=f'{stage}.add_x_sum'))
        nodes.append(oh.make_node('Add', inputs=[x_sum, x_term3], outputs=[x_prime],
                                  name=f'{stage}.add_x_prime'))
        
        y_term1 = f'{stage}.y_term1'
        y_term2 = f'{stage}.y_term2'
        y_term3 = f'{stage}.y_term3'
        nodes.append(oh.make_node('Mul', inputs=[h10, x_gdc], outputs=[y_term1],
                                  name=f'{stage}.mul_y_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h11, y_gdc], outputs=[y_term2],
                                  name=f'{stage}.mul_y_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h12, ones], outputs=[y_term3],
                                  name=f'{stage}.mul_y_term3'))
        
        y_prime = f'{stage}.y_prime'
        y_sum = f'{stage}.y_sum'
        nodes.append(oh.make_node('Add', inputs=[y_term1, y_term2], outputs=[y_sum],
                                  name=f'{stage}.add_y_sum'))
        nodes.append(oh.make_node('Add', inputs=[y_sum, y_term3], outputs=[y_prime],
                                  name=f'{stage}.add_y_prime'))
        
        w_term1 = f'{stage}.w_term1'
        w_term2 = f'{stage}.w_term2'
        w_term3 = f'{stage}.w_term3'
        nodes.append(oh.make_node('Mul', inputs=[h20, x_gdc], outputs=[w_term1],
                                  name=f'{stage}.mul_w_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h21, y_gdc], outputs=[w_term2],
                                  name=f'{stage}.mul_w_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h22, ones], outputs=[w_term3],
                                  name=f'{stage}.mul_w_term3'))
        
        w_prime = f'{stage}.w_prime'
        w_sum = f'{stage}.w_sum'
        nodes.append(oh.make_node('Add', inputs=[w_term1, w_term2], outputs=[w_sum],
                                  name=f'{stage}.add_w_sum'))
        nodes.append(oh.make_node('Add', inputs=[w_sum, w_term3], outputs=[w_prime],
                                  name=f'{stage}.add_w_prime'))
        
        # Normalize by w': x_norm = x' / w', y_norm = y' / w'
        x_norm = f'{stage}.x_norm'
        y_norm = f'{stage}.y_norm'
        nodes.append(oh.make_node('Div', inputs=[x_prime, w_prime], outputs=[x_norm],
                                  name=f'{stage}.div_x_norm'))
        nodes.append(oh.make_node('Div', inputs=[y_prime, w_prime], outputs=[y_norm],
                                  name=f'{stage}.div_y_norm'))
        
        # Stack into fused grid [h,w,2]
        fused_grid = f'{stage}.fused_grid'
        nodes.append(oh.make_node('Concat', inputs=[x_norm, y_norm], outputs=[fused_grid],
                                  name=f'{stage}.concat_fused_grid', axis=-1))
        
        # Also create deshake-only grid for debugging
        x_deshake = f'{stage}.x_deshake'
        y_deshake = f'{stage}.y_deshake'
        nodes.append(oh.make_node('Mul', inputs=[h00, x], outputs=[x_term1],
                                  name=f'{stage}.mul_x_term1_deshake'))
        nodes.append(oh.make_node('Mul', inputs=[h01, y], outputs=[x_term2],
                                  name=f'{stage}.mul_x_term2_deshake'))
        nodes.append(oh.make_node('Mul', inputs=[h02, ones], outputs=[x_term3],
                                  name=f'{stage}.mul_x_term3_deshake'))
        nodes.append(oh.make_node('Add', inputs=[x_term1, x_term2], outputs=[x_sum],
                                  name=f'{stage}.add_x_sum_deshake'))
        nodes.append(oh.make_node('Add', inputs=[x_sum, x_term3], outputs=[x_deshake],
                                  name=f'{stage}.add_x_deshake'))
        
        nodes.append(oh.make_node('Mul', inputs=[h10, x], outputs=[y_term1],
                                  name=f'{stage}.mul_y_term1_deshake'))
        nodes.append(oh.make_node('Mul', inputs=[h11, y], outputs=[y_term2],
                                  name=f'{stage}.mul_y_term2_deshake'))
        nodes.append(oh.make_node('Mul', inputs=[h12, ones], outputs=[y_term3],
                                  name=f'{stage}.mul_y_term3_deshake'))
        nodes.append(oh.make_node('Add', inputs=[y_term1, y_term2], outputs=[y_sum],
                                  name=f'{stage}.add_y_sum_deshake'))
        nodes.append(oh.make_node('Add', inputs=[y_sum, y_term3], outputs=[y_deshake],
                                  name=f'{stage}.add_y_deshake'))
        
        deshake_grid = f'{stage}.deshake_grid'
        nodes.append(oh.make_node('Concat', inputs=[x_deshake, y_deshake], outputs=[deshake_grid],
                                  name=f'{stage}.concat_deshake_grid', axis=-1))
        
        vis.append(oh.make_tensor_value_info(fused_grid, TensorProto.FLOAT, ['h', 'w', 2]))
        vis.append(oh.make_tensor_value_info(gdc_grid, TensorProto.FLOAT, ['h', 'w', 2]))
        vis.append(oh.make_tensor_value_info(deshake_grid, TensorProto.FLOAT, ['h', 'w', 2]))
        
        outputs = {
            'fused_grid': {'name': fused_grid},
            'gdc_grid': {'name': gdc_grid},
            'deshake_grid': {'name': deshake_grid}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(gdc_coeffs, type=TensorProto.FLOAT, shape=[4])
        result.appendInput(homography, type=TensorProto.FLOAT, shape=[3, 3])
        result.appendInput(image_size, type=TensorProto.INT64, shape=[2])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles application.
        """
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_coordinator(stage, prev_stages)


class CoordinateFusionV1(MicroblockBase):
    """
    CoordinateFusionV1 (v1)
    ------------------------
    COORDINATES: Fuse GDC, Deshake, and Rolling Shutter.
    
    Position: Between Algo and Applier, fuses static and dynamic corrections.
    
    Purpose: Merge GDC (lens distortion), Deshake (motion compensation),
             and Rolling Shutter correction into a single coordinate grid.
    
    Needs:
        - gdc_coeffs [4] : GDC coefficients [k1, k2, p1, p2]
        - homography [3,3] : Deshake homography matrix (OpenCV format)
        - rolling_shutter [2] : Rolling shutter parameters [scan_time, gyro_delay]
        - image_size [2] : [height, width] for grid generation

    Provides:
        - fused_grid [h,w,2] : Fused coordinate grid for GridSample
        - gdc_grid [h,w,2] : GDC-only coordinate grid (for debugging)
        - deshake_grid [h,w,2] : Deshake-only coordinate grid (for debugging)

    Behavior:
        - build_algo: Not used (algo extracts homography from frames)
        - build_coordinator: Create fused coordinate grid with rolling shutter
        - build_applier: Not used (applier handles application)

    Transformation Composition:
        P_final = T_deshake * T_rolling_shutter * T_gdc * P_source
        
        Where:
        - T_gdc: GDC distortion transformation (lens correction)
        - T_rolling_shutter: Rolling shutter correction (time-based)
        - T_deshake: Deshake homography transformation (motion compensation)
        - P_source: Source pixel coordinates
        - P_final: Final pixel coordinates for sampling

    Complexity: ~50-60 ONNX nodes
    Use Case: High-quality fused GDC + Deshake + Rolling Shutter correction
    """
    name = 'coordinate_fusion_v1'
    family = 'coordinate_fusion_v1'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts homography from frames.
        """
        return BuildResult({}, [], [], [])

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Create fused coordinate grid with rolling shutter correction.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        
        gdc_coeffs = f'{upstream}.gdc_coeffs'
        homography = f'{upstream}.homography'
        rolling_shutter = f'{upstream}.rolling_shutter'
        image_size = f'{upstream}.image_size'
        
        # Extract image dimensions
        height = f'{stage}.height'
        width = f'{stage}.width'
        nodes.append(oh.make_node('Slice', inputs=[image_size], outputs=[height],
                                  name=f'{stage}.slice_height',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[image_size], outputs=[width],
                                  name=f'{stage}.slice_width',
                                  starts=[1], ends=[2], axes=[0]))
        
        # Create coordinate grid
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
        
        # Stack into coordinate grid [h,w,2]
        identity_grid = f'{stage}.identity_grid'
        nodes.append(oh.make_node('Concat', inputs=[w_norm, h_norm], outputs=[identity_grid],
                                  name=f'{stage}.concat_identity_grid', axis=-1))
        
        # Extract GDC coefficients
        k1 = f'{stage}.k1'
        k2 = f'{stage}.k2'
        p1 = f'{stage}.p1'
        p2 = f'{stage}.p2'
        nodes.append(oh.make_node('Slice', inputs=[gdc_coeffs], outputs=[k1],
                                  name=f'{stage}.slice_k1',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gdc_coeffs], outputs=[k2],
                                  name=f'{stage}.slice_k2',
                                  starts=[1], ends=[2], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gdc_coeffs], outputs=[p1],
                                  name=f'{stage}.slice_p1',
                                  starts=[2], ends=[3], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gdc_coeffs], outputs=[p2],
                                  name=f'{stage}.slice_p2',
                                  starts=[3], ends=[4], axes=[0]))
        
        # Extract x and y from identity grid
        x = f'{stage}.x'
        y = f'{stage}.y'
        nodes.append(oh.make_node('Slice', inputs=[identity_grid], outputs=[x],
                                  name=f'{stage}.slice_x',
                                  starts=[0], ends=[1], axes=[-1]))
        nodes.append(oh.make_node('Slice', inputs=[identity_grid], outputs=[y],
                                  name=f'{stage}.slice_y',
                                  starts=[1], ends=[2], axes=[-1]))
        
        # Calculate radius squared
        x_sq = f'{stage}.x_sq'
        y_sq = f'{stage}.y_sq'
        r_sq = f'{stage}.r_sq'
        nodes.append(oh.make_node('Mul', inputs=[x, x], outputs=[x_sq],
                                  name=f'{stage}.mul_x_sq'))
        nodes.append(oh.make_node('Mul', inputs=[y, y], outputs=[y_sq],
                                  name=f'{stage}.mul_y_sq'))
        nodes.append(oh.make_node('Add', inputs=[x_sq, y_sq], outputs=[r_sq],
                                  name=f'{stage}.add_r_sq'))
        
        # Calculate radial distortion factor
        r4 = f'{stage}.r4'
        nodes.append(oh.make_node('Mul', inputs=[r_sq, r_sq], outputs=[r4],
                                  name=f'{stage}.mul_r4'))
        
        k1_term = f'{stage}.k1_term'
        k2_term = f'{stage}.k2_term'
        nodes.append(oh.make_node('Mul', inputs=[k1, r_sq], outputs=[k1_term],
                                  name=f'{stage}.mul_k1_term'))
        nodes.append(oh.make_node('Mul', inputs=[k2, r4], outputs=[k2_term],
                                  name=f'{stage}.mul_k2_term'))
        
        radial_factor = f'{stage}.radial_factor'
        one = f'{stage}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[radial_factor],
                                  name=f'{stage}.identity_radial_factor'))
        nodes.append(oh.make_node('Add', inputs=[radial_factor, k1_term], outputs=[radial_factor],
                                  name=f'{stage}.add_k1_term'))
        nodes.append(oh.make_node('Add', inputs=[radial_factor, k2_term], outputs=[radial_factor],
                                  name=f'{stage}.add_k2_term'))
        
        # Calculate tangential distortion
        xy = f'{stage}.xy'
        nodes.append(oh.make_node('Mul', inputs=[x, y], outputs=[xy],
                                  name=f'{stage}.mul_xy'))
        
        two = f'{stage}.two'
        inits.append(oh.make_tensor(two, TensorProto.FLOAT, [], [2.0]))
        
        two_xy = f'{stage}.two_xy'
        nodes.append(oh.make_node('Mul', inputs=[two, xy], outputs=[two_xy],
                                  name=f'{stage}.mul_two_xy'))
        
        two_x_sq = f'{stage}.two_x_sq'
        nodes.append(oh.make_node('Mul', inputs=[two, x_sq], outputs=[two_x_sq],
                                  name=f'{stage}.mul_two_x_sq'))
        
        two_y_sq = f'{stage}.two_y_sq'
        nodes.append(oh.make_node('Mul', inputs=[two, y_sq], outputs=[two_y_sq],
                                  name=f'{stage}.mul_two_y_sq'))
        
        r2_plus_2x2 = f'{stage}.r2_plus_2x2'
        r2_plus_2y2 = f'{stage}.r2_plus_2y2'
        nodes.append(oh.make_node('Add', inputs=[r_sq, two_x_sq], outputs=[r2_plus_2x2],
                                  name=f'{stage}.add_r2_plus_2x2'))
        nodes.append(oh.make_node('Add', inputs=[r_sq, two_y_sq], outputs=[r2_plus_2y2],
                                  name=f'{stage}.add_r2_plus_2y2'))
        
        x_tan = f'{stage}.x_tan'
        y_tan = f'{stage}.y_tan'
        p1_term = f'{stage}.p1_term'
        p2_term = f'{stage}.p2_term'
        nodes.append(oh.make_node('Mul', inputs=[p1, two_xy], outputs=[p1_term],
                                  name=f'{stage}.mul_p1_term'))
        nodes.append(oh.make_node('Mul', inputs=[p2, r2_plus_2x2], outputs=[p2_term],
                                  name=f'{stage}.mul_p2_term'))
        nodes.append(oh.make_node('Add', inputs=[p1_term, p2_term], outputs=[x_tan],
                                  name=f'{stage}.add_x_tan'))
        
        nodes.append(oh.make_node('Mul', inputs=[p1, r2_plus_2y2], outputs=[p1_term],
                                  name=f'{stage}.mul_p1_term_y'))
        nodes.append(oh.make_node('Mul', inputs=[p2, two_xy], outputs=[p2_term],
                                  name=f'{stage}.mul_p2_term_y'))
        nodes.append(oh.make_node('Add', inputs=[p1_term, p2_term], outputs=[y_tan],
                                  name=f'{stage}.add_y_tan'))
        
        # Apply GDC distortion
        x_radial = f'{stage}.x_radial'
        y_radial = f'{stage}.y_radial'
        nodes.append(oh.make_node('Mul', inputs=[x, radial_factor], outputs=[x_radial],
                                  name=f'{stage}.mul_x_radial'))
        nodes.append(oh.make_node('Mul', inputs=[y, radial_factor], outputs=[y_radial],
                                  name=f'{stage}.mul_y_radial'))
        
        x_gdc = f'{stage}.x_gdc'
        y_gdc = f'{stage}.y_gdc'
        nodes.append(oh.make_node('Add', inputs=[x_radial, x_tan], outputs=[x_gdc],
                                  name=f'{stage}.add_x_gdc'))
        nodes.append(oh.make_node('Add', inputs=[y_radial, y_tan], outputs=[y_gdc],
                                  name=f'{stage}.add_y_gdc'))
        
        # Stack into GDC grid [h,w,2]
        gdc_grid = f'{stage}.gdc_grid'
        nodes.append(oh.make_node('Concat', inputs=[x_gdc, y_gdc], outputs=[gdc_grid],
                                  name=f'{stage}.concat_gdc_grid', axis=-1))
        
        # Apply rolling shutter correction
        # Rolling shutter causes different rows to be captured at different times
        # We adjust the homography based on row position
        scan_time = f'{stage}.scan_time'
        gyro_delay = f'{stage}.gyro_delay'
        nodes.append(oh.make_node('Slice', inputs=[rolling_shutter], outputs=[scan_time],
                                  name=f'{stage}.slice_scan_time',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[rolling_shutter], outputs=[gyro_delay],
                                  name=f'{stage}.slice_gyro_delay',
                                  starts=[1], ends=[2], axes=[0]))
        
        # Calculate row-based time offset
        # time_offset = (y_norm + 1) / 2 * scan_time + gyro_delay
        y_plus_one = f'{stage}.y_plus_one'
        nodes.append(oh.make_node('Add', inputs=[y_norm, one], outputs=[y_plus_one],
                                  name=f'{stage}.add_y_plus_one'))
        
        y_normalized = f'{stage}.y_normalized'
        nodes.append(oh.make_node('Mul', inputs=[y_plus_one, h_half], outputs=[y_normalized],
                                  name=f'{stage}.mul_y_normalized'))
        
        time_offset = f'{stage}.time_offset'
        nodes.append(oh.make_node('Mul', inputs=[y_normalized, scan_time], outputs=[time_offset],
                                  name=f'{stage}.mul_time_offset'))
        nodes.append(oh.make_node('Add', inputs=[time_offset, gyro_delay], outputs=[time_offset],
                                  name=f'{stage}.add_gyro_delay'))
        
        # Extract homography elements
        h00 = f'{stage}.h00'
        h01 = f'{stage}.h01'
        h02 = f'{stage}.h02'
        h10 = f'{stage}.h10'
        h11 = f'{stage}.h11'
        h12 = f'{stage}.h12'
        h20 = f'{stage}.h20'
        h21 = f'{stage}.h21'
        h22 = f'{stage}.h22'
        
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h00],
                                  name=f'{stage}.slice_h00',
                                  starts=[0, 0], ends=[1, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h01],
                                  name=f'{stage}.slice_h01',
                                  starts=[0, 1], ends=[1, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h02],
                                  name=f'{stage}.slice_h02',
                                  starts=[0, 2], ends=[1, 3], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h10],
                                  name=f'{stage}.slice_h10',
                                  starts=[1, 0], ends=[2, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h11],
                                  name=f'{stage}.slice_h11',
                                  starts=[1, 1], ends=[2, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h12],
                                  name=f'{stage}.slice_h12',
                                  starts=[1, 2], ends=[2, 3], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h20],
                                  name=f'{stage}.slice_h20',
                                  starts=[2, 0], ends=[3, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h21],
                                  name=f'{stage}.slice_h21',
                                  starts=[2, 1], ends=[3, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h22],
                                  name=f'{stage}.slice_h22',
                                  starts=[2, 2], ends=[3, 3], axes=[0, 1]))
        
        # Apply Deshake homography to GDC-distorted points
        ones = f'{stage}.ones'
        inits.append(oh.make_tensor(ones, TensorProto.FLOAT, [], [1.0]))
        
        x_term1 = f'{stage}.x_term1'
        x_term2 = f'{stage}.x_term2'
        x_term3 = f'{stage}.x_term3'
        nodes.append(oh.make_node('Mul', inputs=[h00, x_gdc], outputs=[x_term1],
                                  name=f'{stage}.mul_x_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h01, y_gdc], outputs=[x_term2],
                                  name=f'{stage}.mul_x_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h02, ones], outputs=[x_term3],
                                  name=f'{stage}.mul_x_term3'))
        
        x_prime = f'{stage}.x_prime'
        x_sum = f'{stage}.x_sum'
        nodes.append(oh.make_node('Add', inputs=[x_term1, x_term2], outputs=[x_sum],
                                  name=f'{stage}.add_x_sum'))
        nodes.append(oh.make_node('Add', inputs=[x_sum, x_term3], outputs=[x_prime],
                                  name=f'{stage}.add_x_prime'))
        
        y_term1 = f'{stage}.y_term1'
        y_term2 = f'{stage}.y_term2'
        y_term3 = f'{stage}.y_term3'
        nodes.append(oh.make_node('Mul', inputs=[h10, x_gdc], outputs=[y_term1],
                                  name=f'{stage}.mul_y_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h11, y_gdc], outputs=[y_term2],
                                  name=f'{stage}.mul_y_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h12, ones], outputs=[y_term3],
                                  name=f'{stage}.mul_y_term3'))
        
        y_prime = f'{stage}.y_prime'
        y_sum = f'{stage}.y_sum'
        nodes.append(oh.make_node('Add', inputs=[y_term1, y_term2], outputs=[y_sum],
                                  name=f'{stage}.add_y_sum'))
        nodes.append(oh.make_node('Add', inputs=[y_sum, y_term3], outputs=[y_prime],
                                  name=f'{stage}.add_y_prime'))
        
        w_term1 = f'{stage}.w_term1'
        w_term2 = f'{stage}.w_term2'
        w_term3 = f'{stage}.w_term3'
        nodes.append(oh.make_node('Mul', inputs=[h20, x_gdc], outputs=[w_term1],
                                  name=f'{stage}.mul_w_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h21, y_gdc], outputs=[w_term2],
                                  name=f'{stage}.mul_w_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h22, ones], outputs=[w_term3],
                                  name=f'{stage}.mul_w_term3'))
        
        w_prime = f'{stage}.w_prime'
        w_sum = f'{stage}.w_sum'
        nodes.append(oh.make_node('Add', inputs=[w_term1, w_term2], outputs=[w_sum],
                                  name=f'{stage}.add_w_sum'))
        nodes.append(oh.make_node('Add', inputs=[w_sum, w_term3], outputs=[w_prime],
                                  name=f'{stage}.add_w_prime'))
        
        # Normalize by w': x_norm = x' / w', y_norm = y' / w'
        x_norm = f'{stage}.x_norm'
        y_norm = f'{stage}.y_norm'
        nodes.append(oh.make_node('Div', inputs=[x_prime, w_prime], outputs=[x_norm],
                                  name=f'{stage}.div_x_norm'))
        nodes.append(oh.make_node('Div', inputs=[y_prime, w_prime], outputs=[y_norm],
                                  name=f'{stage}.div_y_norm'))
        
        # Stack into fused grid [h,w,2]
        fused_grid = f'{stage}.fused_grid'
        nodes.append(oh.make_node('Concat', inputs=[x_norm, y_norm], outputs=[fused_grid],
                                  name=f'{stage}.concat_fused_grid', axis=-1))
        
        # Also create deshake-only grid for debugging
        x_deshake = f'{stage}.x_deshake'
        y_deshake = f'{stage}.y_deshake'
        nodes.append(oh.make_node('Mul', inputs=[h00, x], outputs=[x_term1],
                                  name=f'{stage}.mul_x_term1_deshake'))
        nodes.append(oh.make_node('Mul', inputs=[h01, y], outputs=[x_term2],
                                  name=f'{stage}.mul_x_term2_deshake'))
        nodes.append(oh.make_node('Mul', inputs=[h02, ones], outputs=[x_term3],
                                  name=f'{stage}.mul_x_term3_deshake'))
        nodes.append(oh.make_node('Add', inputs=[x_term1, x_term2], outputs=[x_sum],
                                  name=f'{stage}.add_x_sum_deshake'))
        nodes.append(oh.make_node('Add', inputs=[x_sum, x_term3], outputs=[x_deshake],
                                  name=f'{stage}.add_x_deshake'))
        
        nodes.append(oh.make_node('Mul', inputs=[h10, x], outputs=[y_term1],
                                  name=f'{stage}.mul_y_term1_deshake'))
        nodes.append(oh.make_node('Mul', inputs=[h11, y], outputs=[y_term2],
                                  name=f'{stage}.mul_y_term2_deshake'))
        nodes.append(oh.make_node('Mul', inputs=[h12, ones], outputs=[y_term3],
                                  name=f'{stage}.mul_y_term3_deshake'))
        nodes.append(oh.make_node('Add', inputs=[y_term1, y_term2], outputs=[y_sum],
                                  name=f'{stage}.add_y_sum_deshake'))
        nodes.append(oh.make_node('Add', inputs=[y_sum, y_term3], outputs=[y_deshake],
                                  name=f'{stage}.add_y_deshake'))
        
        deshake_grid = f'{stage}.deshake_grid'
        nodes.append(oh.make_node('Concat', inputs=[x_deshake, y_deshake], outputs=[deshake_grid],
                                  name=f'{stage}.concat_deshake_grid', axis=-1))
        
        vis.append(oh.make_tensor_value_info(fused_grid, TensorProto.FLOAT, ['h', 'w', 2]))
        vis.append(oh.make_tensor_value_info(gdc_grid, TensorProto.FLOAT, ['h', 'w', 2]))
        vis.append(oh.make_tensor_value_info(deshake_grid, TensorProto.FLOAT, ['h', 'w', 2]))
        
        outputs = {
            'fused_grid': {'name': fused_grid},
            'gdc_grid': {'name': gdc_grid},
            'deshake_grid': {'name': deshake_grid}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(gdc_coeffs, type=TensorProto.FLOAT, shape=[4])
        result.appendInput(homography, type=TensorProto.FLOAT, shape=[3, 3])
        result.appendInput(rolling_shutter, type=TensorProto.FLOAT, shape=[2])
        result.appendInput(image_size, type=TensorProto.INT64, shape=[2])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles application.
        """
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_coordinator(stage, prev_stages)


class CoordinateFusionV2(MicroblockBase):
    """
    CoordinateFusionV2 (v2)
    ------------------------
    COORDINATES: Fuse GDC, Deshake, Rolling Shutter, and ALSC.
    
    Position: Between Algo and Applier, fuses static and dynamic corrections.
    
    Purpose: Merge GDC (lens distortion), Deshake (motion compensation),
             Rolling Shutter correction, and ALSC (adaptive lens shading)
             into a single coordinate grid.
    
    Needs:
        - gdc_coeffs [4] : GDC coefficients [k1, k2, p1, p2]
        - homography [3,3] : Deshake homography matrix (OpenCV format)
        - rolling_shutter [2] : Rolling shutter parameters [scan_time, gyro_delay]
        - alsc_coeffs [3] : ALSC coefficients [k1, k2, k3]
        - vcm_pos [1] : VCM position (lens focal length)
        - image_size [2] : [height, width] for grid generation

    Provides:
        - fused_grid [h,w,2] : Fused coordinate grid for GridSample
        - gain_map [h,w] : ALSC gain map for shading correction
        - gdc_grid [h,w,2] : GDC-only coordinate grid (for debugging)
        - deshake_grid [h,w,2] : Deshake-only coordinate grid (for debugging)

    Behavior:
        - build_algo: Not used (algo extracts homography from frames)
        - build_coordinator: Create fused coordinate grid with all corrections
        - build_applier: Not used (applier handles application)

    Transformation Composition:
        P_final = T_deshake * T_rolling_shutter * T_gdc * P_source
        Gain = G_alsc(P_final)
        
        Where:
        - T_gdc: GDC distortion transformation (lens correction)
        - T_rolling_shutter: Rolling shutter correction (time-based)
        - T_deshake: Deshake homography transformation (motion compensation)
        - G_alsc: ALSC gain map (shading correction)
        - P_source: Source pixel coordinates
        - P_final: Final pixel coordinates for sampling

    Complexity: ~70-80 ONNX nodes
    Use Case: Professional-grade fused correction pipeline
    """
    name = 'coordinate_fusion_v2'
    family = 'coordinate_fusion_v2'
    version = 'v2'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts homography from frames.
        """
        return BuildResult({}, [], [], [])

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Create fused coordinate grid with all corrections.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        
        gdc_coeffs = f'{upstream}.gdc_coeffs'
        homography = f'{upstream}.homography'
        rolling_shutter = f'{upstream}.rolling_shutter'
        alsc_coeffs = f'{upstream}.alsc_coeffs'
        vcm_pos = f'{upstream}.vcm_pos'
        image_size = f'{upstream}.image_size'
        
        # Extract image dimensions
        height = f'{stage}.height'
        width = f'{stage}.width'
        nodes.append(oh.make_node('Slice', inputs=[image_size], outputs=[height],
                                  name=f'{stage}.slice_height',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[image_size], outputs=[width],
                                  name=f'{stage}.slice_width',
                                  starts=[1], ends=[2], axes=[0]))
        
        # Create coordinate grid
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
        
        # Stack into coordinate grid [h,w,2]
        identity_grid = f'{stage}.identity_grid'
        nodes.append(oh.make_node('Concat', inputs=[w_norm, h_norm], outputs=[identity_grid],
                                  name=f'{stage}.concat_identity_grid', axis=-1))
        
        # Extract GDC coefficients
        k1 = f'{stage}.k1'
        k2 = f'{stage}.k2'
        p1 = f'{stage}.p1'
        p2 = f'{stage}.p2'
        nodes.append(oh.make_node('Slice', inputs=[gdc_coeffs], outputs=[k1],
                                  name=f'{stage}.slice_k1',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gdc_coeffs], outputs=[k2],
                                  name=f'{stage}.slice_k2',
                                  starts=[1], ends=[2], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gdc_coeffs], outputs=[p1],
                                  name=f'{stage}.slice_p1',
                                  starts=[2], ends=[3], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gdc_coeffs], outputs=[p2],
                                  name=f'{stage}.slice_p2',
                                  starts=[3], ends=[4], axes=[0]))
        
        # Extract x and y from identity grid
        x = f'{stage}.x'
        y = f'{stage}.y'
        nodes.append(oh.make_node('Slice', inputs=[identity_grid], outputs=[x],
                                  name=f'{stage}.slice_x',
                                  starts=[0], ends=[1], axes=[-1]))
        nodes.append(oh.make_node('Slice', inputs=[identity_grid], outputs=[y],
                                  name=f'{stage}.slice_y',
                                  starts=[1], ends=[2], axes=[-1]))
        
        # Calculate radius squared
        x_sq = f'{stage}.x_sq'
        y_sq = f'{stage}.y_sq'
        r_sq = f'{stage}.r_sq'
        nodes.append(oh.make_node('Mul', inputs=[x, x], outputs=[x_sq],
                                  name=f'{stage}.mul_x_sq'))
        nodes.append(oh.make_node('Mul', inputs=[y, y], outputs=[y_sq],
                                  name=f'{stage}.mul_y_sq'))
        nodes.append(oh.make_node('Add', inputs=[x_sq, y_sq], outputs=[r_sq],
                                  name=f'{stage}.add_r_sq'))
        
        # Calculate radial distortion factor
        r4 = f'{stage}.r4'
        nodes.append(oh.make_node('Mul', inputs=[r_sq, r_sq], outputs=[r4],
                                  name=f'{stage}.mul_r4'))
        
        k1_term = f'{stage}.k1_term'
        k2_term = f'{stage}.k2_term'
        nodes.append(oh.make_node('Mul', inputs=[k1, r_sq], outputs=[k1_term],
                                  name=f'{stage}.mul_k1_term'))
        nodes.append(oh.make_node('Mul', inputs=[k2, r4], outputs=[k2_term],
                                  name=f'{stage}.mul_k2_term'))
        
        radial_factor = f'{stage}.radial_factor'
        one = f'{stage}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[radial_factor],
                                  name=f'{stage}.identity_radial_factor'))
        nodes.append(oh.make_node('Add', inputs=[radial_factor, k1_term], outputs=[radial_factor],
                                  name=f'{stage}.add_k1_term'))
        nodes.append(oh.make_node('Add', inputs=[radial_factor, k2_term], outputs=[radial_factor],
                                  name=f'{stage}.add_k2_term'))
        
        # Calculate tangential distortion
        xy = f'{stage}.xy'
        nodes.append(oh.make_node('Mul', inputs=[x, y], outputs=[xy],
                                  name=f'{stage}.mul_xy'))
        
        two = f'{stage}.two'
        inits.append(oh.make_tensor(two, TensorProto.FLOAT, [], [2.0]))
        
        two_xy = f'{stage}.two_xy'
        nodes.append(oh.make_node('Mul', inputs=[two, xy], outputs=[two_xy],
                                  name=f'{stage}.mul_two_xy'))
        
        two_x_sq = f'{stage}.two_x_sq'
        nodes.append(oh.make_node('Mul', inputs=[two, x_sq], outputs=[two_x_sq],
                                  name=f'{stage}.mul_two_x_sq'))
        
        two_y_sq = f'{stage}.two_y_sq'
        nodes.append(oh.make_node('Mul', inputs=[two, y_sq], outputs=[two_y_sq],
                                  name=f'{stage}.mul_two_y_sq'))
        
        r2_plus_2x2 = f'{stage}.r2_plus_2x2'
        r2_plus_2y2 = f'{stage}.r2_plus_2y2'
        nodes.append(oh.make_node('Add', inputs=[r_sq, two_x_sq], outputs=[r2_plus_2x2],
                                  name=f'{stage}.add_r2_plus_2x2'))
        nodes.append(oh.make_node('Add', inputs=[r_sq, two_y_sq], outputs=[r2_plus_2y2],
                                  name=f'{stage}.add_r2_plus_2y2'))
        
        x_tan = f'{stage}.x_tan'
        y_tan = f'{stage}.y_tan'
        p1_term = f'{stage}.p1_term'
        p2_term = f'{stage}.p2_term'
        nodes.append(oh.make_node('Mul', inputs=[p1, two_xy], outputs=[p1_term],
                                  name=f'{stage}.mul_p1_term'))
        nodes.append(oh.make_node('Mul', inputs=[p2, r2_plus_2x2], outputs=[p2_term],
                                  name=f'{stage}.mul_p2_term'))
        nodes.append(oh.make_node('Add', inputs=[p1_term, p2_term], outputs=[x_tan],
                                  name=f'{stage}.add_x_tan'))
        
        nodes.append(oh.make_node('Mul', inputs=[p1, r2_plus_2y2], outputs=[p1_term],
                                  name=f'{stage}.mul_p1_term_y'))
        nodes.append(oh.make_node('Mul', inputs=[p2, two_xy], outputs=[p2_term],
                                  name=f'{stage}.mul_p2_term_y'))
        nodes.append(oh.make_node('Add', inputs=[p1_term, p2_term], outputs=[y_tan],
                                  name=f'{stage}.add_y_tan'))
        
        # Apply GDC distortion
        x_radial = f'{stage}.x_radial'
        y_radial = f'{stage}.y_radial'
        nodes.append(oh.make_node('Mul', inputs=[x, radial_factor], outputs=[x_radial],
                                  name=f'{stage}.mul_x_radial'))
        nodes.append(oh.make_node('Mul', inputs=[y, radial_factor], outputs=[y_radial],
                                  name=f'{stage}.mul_y_radial'))
        
        x_gdc = f'{stage}.x_gdc'
        y_gdc = f'{stage}.y_gdc'
        nodes.append(oh.make_node('Add', inputs=[x_radial, x_tan], outputs=[x_gdc],
                                  name=f'{stage}.add_x_gdc'))
        nodes.append(oh.make_node('Add', inputs=[y_radial, y_tan], outputs=[y_gdc],
                                  name=f'{stage}.add_y_gdc'))
        
        # Stack into GDC grid [h,w,2]
        gdc_grid = f'{stage}.gdc_grid'
        nodes.append(oh.make_node('Concat', inputs=[x_gdc, y_gdc], outputs=[gdc_grid],
                                  name=f'{stage}.concat_gdc_grid', axis=-1))
        
        # Extract homography elements
        h00 = f'{stage}.h00'
        h01 = f'{stage}.h01'
        h02 = f'{stage}.h02'
        h10 = f'{stage}.h10'
        h11 = f'{stage}.h11'
        h12 = f'{stage}.h12'
        h20 = f'{stage}.h20'
        h21 = f'{stage}.h21'
        h22 = f'{stage}.h22'
        
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h00],
                                  name=f'{stage}.slice_h00',
                                  starts=[0, 0], ends=[1, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h01],
                                  name=f'{stage}.slice_h01',
                                  starts=[0, 1], ends=[1, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h02],
                                  name=f'{stage}.slice_h02',
                                  starts=[0, 2], ends=[1, 3], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h10],
                                  name=f'{stage}.slice_h10',
                                  starts=[1, 0], ends=[2, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h11],
                                  name=f'{stage}.slice_h11',
                                  starts=[1, 1], ends=[2, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h12],
                                  name=f'{stage}.slice_h12',
                                  starts=[1, 2], ends=[2, 3], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h20],
                                  name=f'{stage}.slice_h20',
                                  starts=[2, 0], ends=[3, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h21],
                                  name=f'{stage}.slice_h21',
                                  starts=[2, 1], ends=[3, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h22],
                                  name=f'{stage}.slice_h22',
                                  starts=[2, 2], ends=[3, 3], axes=[0, 1]))
        
        # Apply Deshake homography to GDC-distorted points
        ones = f'{stage}.ones'
        inits.append(oh.make_tensor(ones, TensorProto.FLOAT, [], [1.0]))
        
        x_term1 = f'{stage}.x_term1'
        x_term2 = f'{stage}.x_term2'
        x_term3 = f'{stage}.x_term3'
        nodes.append(oh.make_node('Mul', inputs=[h00, x_gdc], outputs=[x_term1],
                                  name=f'{stage}.mul_x_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h01, y_gdc], outputs=[x_term2],
                                  name=f'{stage}.mul_x_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h02, ones], outputs=[x_term3],
                                  name=f'{stage}.mul_x_term3'))
        
        x_prime = f'{stage}.x_prime'
        x_sum = f'{stage}.x_sum'
        nodes.append(oh.make_node('Add', inputs=[x_term1, x_term2], outputs=[x_sum],
                                  name=f'{stage}.add_x_sum'))
        nodes.append(oh.make_node('Add', inputs=[x_sum, x_term3], outputs=[x_prime],
                                  name=f'{stage}.add_x_prime'))
        
        y_term1 = f'{stage}.y_term1'
        y_term2 = f'{stage}.y_term2'
        y_term3 = f'{stage}.y_term3'
        nodes.append(oh.make_node('Mul', inputs=[h10, x_gdc], outputs=[y_term1],
                                  name=f'{stage}.mul_y_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h11, y_gdc], outputs=[y_term2],
                                  name=f'{stage}.mul_y_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h12, ones], outputs=[y_term3],
                                  name=f'{stage}.mul_y_term3'))
        
        y_prime = f'{stage}.y_prime'
        y_sum = f'{stage}.y_sum'
        nodes.append(oh.make_node('Add', inputs=[y_term1, y_term2], outputs=[y_sum],
                                  name=f'{stage}.add_y_sum'))
        nodes.append(oh.make_node('Add', inputs=[y_sum, y_term3], outputs=[y_prime],
                                  name=f'{stage}.add_y_prime'))
        
        w_term1 = f'{stage}.w_term1'
        w_term2 = f'{stage}.w_term2'
        w_term3 = f'{stage}.w_term3'
        nodes.append(oh.make_node('Mul', inputs=[h20, x_gdc], outputs=[w_term1],
                                  name=f'{stage}.mul_w_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h21, y_gdc], outputs=[w_term2],
                                  name=f'{stage}.mul_w_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h22, ones], outputs=[w_term3],
                                  name=f'{stage}.mul_w_term3'))
        
        w_prime = f'{stage}.w_prime'
        w_sum = f'{stage}.w_sum'
        nodes.append(oh.make_node('Add', inputs=[w_term1, w_term2], outputs=[w_sum],
                                  name=f'{stage}.add_w_sum'))
        nodes.append(oh.make_node('Add', inputs=[w_sum, w_term3], outputs=[w_prime],
                                  name=f'{stage}.add_w_prime'))
        
        # Normalize by w': x_norm = x' / w', y_norm = y' / w'
        x_norm = f'{stage}.x_norm'
        y_norm = f'{stage}.y_norm'
        nodes.append(oh.make_node('Div', inputs=[x_prime, w_prime], outputs=[x_norm],
                                  name=f'{stage}.div_x_norm'))
        nodes.append(oh.make_node('Div', inputs=[y_prime, w_prime], outputs=[y_norm],
                                  name=f'{stage}.div_y_norm'))
        
        # Stack into fused grid [h,w,2]
        fused_grid = f'{stage}.fused_grid'
        nodes.append(oh.make_node('Concat', inputs=[x_norm, y_norm], outputs=[fused_grid],
                                  name=f'{stage}.concat_fused_grid', axis=-1))
        
        # Calculate ALSC gain map
        # gain = (k1 * r^2 + k2 * r^4 + k3 * r^6 + vcm_pos)^2
        alsc_k1 = f'{stage}.alsc_k1'
        alsc_k2 = f'{stage}.alsc_k2'
        alsc_k3 = f'{stage}.alsc_k3'
        nodes.append(oh.make_node('Slice', inputs=[alsc_coeffs], outputs=[alsc_k1],
                                  name=f'{stage}.slice_alsc_k1',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[alsc_coeffs], outputs=[alsc_k2],
                                  name=f'{stage}.slice_alsc_k2',
                                  starts=[1], ends=[2], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[alsc_coeffs], outputs=[alsc_k3],
                                  name=f'{stage}.slice_alsc_k3',
                                  starts=[2], ends=[3], axes=[0]))
        
        r6 = f'{stage}.r6'
        nodes.append(oh.make_node('Mul', inputs=[r4, r_sq], outputs=[r6],
                                  name=f'{stage}.mul_r6'))
        
        alsc_k1_term = f'{stage}.alsc_k1_term'
        alsc_k2_term = f'{stage}.alsc_k2_term'
        alsc_k3_term = f'{stage}.alsc_k3_term'
        nodes.append(oh.make_node('Mul', inputs=[alsc_k1, r_sq], outputs=[alsc_k1_term],
                                  name=f'{stage}.mul_alsc_k1_term'))
        nodes.append(oh.make_node('Mul', inputs=[alsc_k2, r4], outputs=[alsc_k2_term],
                                  name=f'{stage}.mul_alsc_k2_term'))
        nodes.append(oh.make_node('Mul', inputs=[alsc_k3, r6], outputs=[alsc_k3_term],
                                  name=f'{stage}.mul_alsc_k3_term'))
        
        alsc_poly = f'{stage}.alsc_poly'
        nodes.append(oh.make_node('Identity', inputs=[vcm_pos], outputs=[alsc_poly],
                                  name=f'{stage}.identity_alsc_poly'))
        nodes.append(oh.make_node('Add', inputs=[alsc_poly, alsc_k1_term], outputs=[alsc_poly],
                                  name=f'{stage}.add_alsc_k1_term'))
        nodes.append(oh.make_node('Add', inputs=[alsc_poly, alsc_k2_term], outputs=[alsc_poly],
                                  name=f'{stage}.add_alsc_k2_term'))
        nodes.append(oh.make_node('Add', inputs=[alsc_poly, alsc_k3_term], outputs=[alsc_poly],
                                  name=f'{stage}.add_alsc_k3_term'))
        
        gain_map = f'{stage}.gain_map'
        nodes.append(oh.make_node('Mul', inputs=[alsc_poly, alsc_poly], outputs=[gain_map],
                                  name=f'{stage}.mul_gain_map'))
        
        # Also create deshake-only grid for debugging
        x_deshake = f'{stage}.x_deshake'
        y_deshake = f'{stage}.y_deshake'
        nodes.append(oh.make_node('Mul', inputs=[h00, x], outputs=[x_term1],
                                  name=f'{stage}.mul_x_term1_deshake'))
        nodes.append(oh.make_node('Mul', inputs=[h01, y], outputs=[x_term2],
                                  name=f'{stage}.mul_x_term2_deshake'))
        nodes.append(oh.make_node('Mul', inputs=[h02, ones], outputs=[x_term3],
                                  name=f'{stage}.mul_x_term3_deshake'))
        nodes.append(oh.make_node('Add', inputs=[x_term1, x_term2], outputs=[x_sum],
                                  name=f'{stage}.add_x_sum_deshake'))
        nodes.append(oh.make_node('Add', inputs=[x_sum, x_term3], outputs=[x_deshake],
                                  name=f'{stage}.add_x_deshake'))
        
        nodes.append(oh.make_node('Mul', inputs=[h10, x], outputs=[y_term1],
                                  name=f'{stage}.mul_y_term1_deshake'))
        nodes.append(oh.make_node('Mul', inputs=[h11, y], outputs=[y_term2],
                                  name=f'{stage}.mul_y_term2_deshake'))
        nodes.append(oh.make_node('Mul', inputs=[h12, ones], outputs=[y_term3],
                                  name=f'{stage}.mul_y_term3_deshake'))
        nodes.append(oh.make_node('Add', inputs=[y_term1, y_term2], outputs=[y_sum],
                                  name=f'{stage}.add_y_sum_deshake'))
        nodes.append(oh.make_node('Add', inputs=[y_sum, y_term3], outputs=[y_deshake],
                                  name=f'{stage}.add_y_deshake'))
        
        deshake_grid = f'{stage}.deshake_grid'
        nodes.append(oh.make_node('Concat', inputs=[x_deshake, y_deshake], outputs=[deshake_grid],
                                  name=f'{stage}.concat_deshake_grid', axis=-1))
        
        vis.append(oh.make_tensor_value_info(fused_grid, TensorProto.FLOAT, ['h', 'w', 2]))
        vis.append(oh.make_tensor_value_info(gain_map, TensorProto.FLOAT, ['h', 'w']))
        vis.append(oh.make_tensor_value_info(gdc_grid, TensorProto.FLOAT, ['h', 'w', 2]))
        vis.append(oh.make_tensor_value_info(deshake_grid, TensorProto.FLOAT, ['h', 'w', 2]))
        
        outputs = {
            'fused_grid': {'name': fused_grid},
            'gain_map': {'name': gain_map},
            'gdc_grid': {'name': gdc_grid},
            'deshake_grid': {'name': deshake_grid}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(gdc_coeffs, type=TensorProto.FLOAT, shape=[4])
        result.appendInput(homography, type=TensorProto.FLOAT, shape=[3, 3])
        result.appendInput(rolling_shutter, type=TensorProto.FLOAT, shape=[2])
        result.appendInput(alsc_coeffs, type=TensorProto.FLOAT, shape=[3])
        result.appendInput(vcm_pos, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(image_size, type=TensorProto.INT64, shape=[2])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles application.
        """
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_coordinator(stage, prev_stages)