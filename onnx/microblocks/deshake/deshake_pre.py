from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class DeshakePre(MicroblockBase):
    """
    DeshakePre - Pre-processing Stage (Algo Domain)
    ---------------------------------------------------------------
    PRE-PROCESSING: Calculate homography matrix and GDC coefficients.
    
    Position: After YUV/RGB conversion, before core processing.
    
    Purpose: Calculate homography matrix from consecutive frames and
    GDC coefficients from VCM position. This stage is reusable across
    all deshake implementations.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - prev_frame [n,3,h,w] : Previous video frame (YUV or RGB)
        - vcm_pos [1] : VCM position (lens focal length parameter)
        - gdc_params [4] : GDC base parameters [k1_base, k2_base, p1_base, p2_base]

    Provides:
        - homography [3,3] : 3x3 homography matrix (OpenCV format)
        - confidence [1] : Confidence score (0-1)
        - gdc_coeffs [4] : GDC coefficients [k1, k2, p1, p2] (adjusted by VCM)

    Behavior:
        - build_algo: Extracts homography matrix and calculates GDC coefficients
        - build_coordinator: None (not used in pre-processing)
        - build_applier: None (not used in pre-processing)

    Homography Matrix Format (OpenCV):
        H = [h00 h01 h02]
            [h10 h11 h12]
            [h20 h21 h22]
        
        Where:
        - h00, h01, h10, h11: Rotation and scaling
        - h02, h12: Translation (tx, ty)
        - h20, h21: Perspective (shear)
        - h22: Scale (usually 1)

    GDC Coefficient Calculation:
        GDC coefficients vary with VCM position (lens focal length).
        The VCM position changes from min to max during autofocus.
        
        k1 = k1_base * (1 + vcm_pos * vcm_scale)
        k2 = k2_base * (1 + vcm_pos * vcm_scale)
        p1 = p1_base * (1 + vcm_pos * vcm_scale)
        p2 = p2_base * (1 + vcm_pos * vcm_scale)
        
        Where:
        - vcm_pos: VCM position (normalized 0-1)
        - vcm_scale: Scaling factor for VCM influence

    Complexity: ~20-30 ONNX nodes
    Use Case: Pre-processing stage for all deshake implementations
    """
    name = 'deshake_pre'
    family = 'deshake_pre'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Extract homography matrix and calculate GDC coefficients.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        prev_frame = f'{upstream}.prev_frame'
        vcm_pos = f'{upstream}.vcm_pos'
        gdc_params = f'{upstream}.gdc_params'
        
        # Convert to grayscale for motion estimation
        current_gray = f'{stage}.current_gray'
        prev_gray = f'{stage}.prev_gray'
        three = f'{stage}.three'
        inits.append(oh.make_tensor(three, TensorProto.FLOAT, [], [3.0]))
        
        nodes.append(oh.make_node('ReduceMean', inputs=[current_frame], outputs=[current_gray],
                                  name=f'{stage}.reduce_mean_current', axes=[1], keepdims=1))
        nodes.append(oh.make_node('ReduceMean', inputs=[prev_frame], outputs=[prev_gray],
                                  name=f'{stage}.reduce_mean_prev', axes=[1], keepdims=1))
        
        # Calculate difference for motion estimation
        diff = f'{stage}.diff'
        nodes.append(oh.make_node('Sub', inputs=[current_gray, prev_gray], outputs=[diff],
                                  name=f'{stage}.sub_diff'))
        
        # Extract translation (tx, ty) from difference
        diff_mean = f'{stage}.diff_mean'
        nodes.append(oh.make_node('ReduceMean', inputs=[diff], outputs=[diff_mean],
                                  name=f'{stage}.reduce_mean_diff', keepdims=0))
        
        tx = f'{stage}.tx'
        ty = f'{stage}.ty'
        nodes.append(oh.make_node('Identity', inputs=[diff_mean], outputs=[tx],
                                  name=f'{stage}.identity_tx'))
        nodes.append(oh.make_node('Identity', inputs=[diff_mean], outputs=[ty],
                                  name=f'{stage}.identity_ty'))
        
        # Create homography matrix (translation only)
        # H = [1  0  tx]
        #     [0  1  ty]
        #     [0  0   1]
        one = f'{stage}.one'
        zero = f'{stage}.zero'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
        
        # Create homography matrix elements
        h00 = f'{stage}.h00'
        h01 = f'{stage}.h01'
        h02 = f'{stage}.h02'
        h10 = f'{stage}.h10'
        h11 = f'{stage}.h11'
        h12 = f'{stage}.h12'
        h20 = f'{stage}.h20'
        h21 = f'{stage}.h21'
        h22 = f'{stage}.h22'
        
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[h00],
                                  name=f'{stage}.identity_h00'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h01],
                                  name=f'{stage}.identity_h01'))
        nodes.append(oh.make_node('Identity', inputs=[tx], outputs=[h02],
                                  name=f'{stage}.identity_h02'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h10],
                                  name=f'{stage}.identity_h10'))
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[h11],
                                  name=f'{stage}.identity_h11'))
        nodes.append(oh.make_node('Identity', inputs=[ty], outputs=[h12],
                                  name=f'{stage}.identity_h12'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h20],
                                  name=f'{stage}.identity_h20'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h21],
                                  name=f'{stage}.identity_h21'))
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[h22],
                                  name=f'{stage}.identity_h22'))
        
        # Stack into homography matrix [3,3]
        h_row0 = f'{stage}.h_row0'
        h_row1 = f'{stage}.h_row1'
        h_row2 = f'{stage}.h_row2'
        homography = f'{stage}.homography'
        
        nodes.append(oh.make_node('Concat', inputs=[h00, h01, h02], outputs=[h_row0],
                                  name=f'{stage}.concat_h_row0', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[h10, h11, h12], outputs=[h_row1],
                                  name=f'{stage}.concat_h_row1', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[h20, h21, h22], outputs=[h_row2],
                                  name=f'{stage}.concat_h_row2', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[h_row0, h_row1, h_row2], outputs=[homography],
                                  name=f'{stage}.concat_homography', axis=0))
        
        # Calculate confidence score
        diff_abs = f'{stage}.diff_abs'
        nodes.append(oh.make_node('Abs', inputs=[diff_mean], outputs=[diff_abs],
                                  name=f'{stage}.abs_diff'))
        
        confidence = f'{stage}.confidence'
        confidence_threshold = f'{stage}.confidence_threshold'
        inits.append(oh.make_tensor(confidence_threshold, TensorProto.FLOAT, [], [10.0]))
        
        nodes.append(oh.make_node('Sub', inputs=[confidence_threshold, diff_abs], outputs=[confidence],
                                  name=f'{stage}.sub_confidence'))
        nodes.append(oh.make_node('Clip', inputs=[confidence], outputs=[confidence],
                                  name=f'{stage}.clip_confidence', min=0.0, max=1.0))
        
        # Calculate GDC coefficients from VCM position
        # GDC coefficients vary with VCM position (lens focal length)
        vcm_scale = f'{stage}.vcm_scale'
        inits.append(oh.make_tensor(vcm_scale, TensorProto.FLOAT, [], [0.5]))
        
        vcm_factor = f'{stage}.vcm_factor'
        nodes.append(oh.make_node('Mul', inputs=[vcm_pos, vcm_scale], outputs=[vcm_factor],
                                  name=f'{stage}.mul_vcm_factor'))
        vcm_factor_plus_one = f'{stage}.vcm_factor_plus_one'
        nodes.append(oh.make_node('Add', inputs=[vcm_factor, one], outputs=[vcm_factor_plus_one],
                                  name=f'{stage}.add_vcm_factor_plus_one'))
        
        # Extract GDC base parameters
        k1_base = f'{stage}.k1_base'
        k2_base = f'{stage}.k2_base'
        p1_base = f'{stage}.p1_base'
        p2_base = f'{stage}.p2_base'
        nodes.append(oh.make_node('Slice', inputs=[gdc_params], outputs=[k1_base],
                                  name=f'{stage}.slice_k1_base',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gdc_params], outputs=[k2_base],
                                  name=f'{stage}.slice_k2_base',
                                  starts=[1], ends=[2], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gdc_params], outputs=[p1_base],
                                  name=f'{stage}.slice_p1_base',
                                  starts=[2], ends=[3], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gdc_params], outputs=[p2_base],
                                  name=f'{stage}.slice_p2_base',
                                  starts=[3], ends=[4], axes=[0]))
        
        # Calculate GDC coefficients adjusted by VCM position
        k1 = f'{stage}.k1'
        k2 = f'{stage}.k2'
        p1 = f'{stage}.p1'
        p2 = f'{stage}.p2'
        nodes.append(oh.make_node('Mul', inputs=[k1_base, vcm_factor_plus_one], outputs=[k1],
                                  name=f'{stage}.mul_k1'))
        nodes.append(oh.make_node('Mul', inputs=[k2_base, vcm_factor_plus_one], outputs=[k2],
                                  name=f'{stage}.mul_k2'))
        nodes.append(oh.make_node('Mul', inputs=[p1_base, vcm_factor_plus_one], outputs=[p1],
                                  name=f'{stage}.mul_p1'))
        nodes.append(oh.make_node('Mul', inputs=[p2_base, vcm_factor_plus_one], outputs=[p2],
                                  name=f'{stage}.mul_p2'))
        
        # Stack into GDC coefficients [4]
        gdc_coeffs = f'{stage}.gdc_coeffs'
        nodes.append(oh.make_node('Concat', inputs=[k1, k2, p1, p2], outputs=[gdc_coeffs],
                                  name=f'{stage}.concat_gdc_coeffs', axis=0))
        
        vis.append(oh.make_tensor_value_info(homography, TensorProto.FLOAT, [3, 3]))
        vis.append(oh.make_tensor_value_info(confidence, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(gdc_coeffs, TensorProto.FLOAT, [4]))
        
        outputs = {
            'homography': {'name': homography},
            'confidence': {'name': confidence},
            'gdc_coeffs': {'name': gdc_coeffs}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(current_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(prev_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(vcm_pos, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(gdc_params, type=TensorProto.FLOAT, shape=[4])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used in pre-processing stage.
        """
        return super().build_algo(stage, prev_stages)

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used in pre-processing stage.
        """
        return super().build_applier(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)