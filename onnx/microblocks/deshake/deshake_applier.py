from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class DeshakeApplier(MicroblockBase):
    """
    DeshakeApplier - Grid Mapping Applier (Applier Domain)
    ---------------------------------------------------------------
    APPLIER: Apply grid mapping to frame using mesh grid coefficients.
    
    Position: After mesh grid generator, applies grid mapping.
    
    Purpose: Apply grid mapping to frame using mesh grid coefficients
    from coordinator domain.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - mesh_grid [mesh_h,mesh_w,2] : Mesh vertex grid from coordinator
        - image_size [2] : [height, width] for upsampling
        - padding_mode [1] : Padding mode (0=zeros, 1=border, 2=reflect)

    Provides:
        - stabilized_frame [n,3,h,w] : Stabilized frame (YUV or RGB)
        - valid_mask [n,1,h,w] : Valid region mask

    Behavior:
        - build_algo: Not used (algo extracts homography from frames)
        - build_coordinator: Not used (mesh grid generator handles mesh)
        - build_applier: Apply grid mapping using mesh grid coefficients

    Grid Mapping:
        Uses mesh grid coefficients to warp the image. The mesh grid
        contains the sampling coordinates for each pixel.
        
        Benefits:
        - Hardware-accelerated on GPU
        - Single-pass operation
        - Professional-grade quality

    Edge Filling:
        Uses "Reflect" or "Replicate" padding to hide the black bars
        at the edge of the crop.

    Complexity: ~15-20 ONNX nodes
    Use Case: Grid mapping applier
    """
    name = 'deshake_applier'
    family = 'deshake_applier'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts homography from frames.
        """
        return super().build_algo(stage, prev_stages)

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - mesh grid generator handles mesh.
        """
        return super().build_coordinator(stage, prev_stages)

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply grid mapping using mesh grid coefficients.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        mesh_grid = f'{upstream}.mesh_grid'
        image_size = f'{upstream}.image_size'
        padding_mode = f'{upstream}.padding_mode'
        stabilized_frame = f'{stage}.stabilized_frame'
        
        # Extract image dimensions
        height = f'{stage}.height'
        width = f'{stage}.width'
        nodes.append(oh.make_node('Slice', inputs=[image_size], outputs=[height],
                                  name=f'{stage}.slice_height',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[image_size], outputs=[width],
                                  name=f'{stage}.slice_width',
                                  starts=[1], ends=[2], axes=[0]))
        
        # Upsample mesh grid to full resolution
        # Use Resize operator to interpolate mesh vertices
        mesh_grid_expanded = f'{stage}.mesh_grid_expanded'
        
        # Create scales for upsampling
        scales = f'{stage}.scales'
        inits.append(oh.make_tensor(scales, TensorProto.FLOAT, [4], 
                                  [1.0, 1.0, float(height)/16.0, float(width)/16.0]))
        
        nodes.append(oh.make_node('Resize', inputs=[mesh_grid, 'roi', scales],
                                  outputs=[mesh_grid_expanded],
                                  name=f'{stage}.resize_mesh_grid',
                                  mode='linear',
                                  coordinate_transformation_mode='asymmetric'))
        
        # Unsqueeze for GridSample [1,h,w,2]
        grid_expanded = f'{stage}.grid_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[mesh_grid_expanded], outputs=[grid_expanded],
                                  name=f'{stage}.unsqueeze_grid', axes=[0]))
        
        # Determine padding mode
        # 0 = zeros, 1 = border, 2 = reflect
        padding_mode_int = f'{stage}.padding_mode_int'
        nodes.append(oh.make_node('Cast', inputs=[padding_mode], outputs=[padding_mode_int],
                                  name=f'{stage}.cast_padding_mode',
                                  to=TensorProto.INT64))
        
        # Apply GridSample for grid mapping
        # Use reflect padding for edge filling
        nodes.append(oh.make_node('GridSample', inputs=[current_frame, grid_expanded],
                                  outputs=[stabilized_frame],
                                  name=f'{stage}.gridsample', mode='bilinear',
                                  padding_mode='reflection', align_corners=1))
        
        # Create valid mask
        # Pixels where mesh grid coordinates are within [-1, 1] are valid
        x = f'{stage}.x'
        y = f'{stage}.y'
        nodes.append(oh.make_node('Slice', inputs=[mesh_grid_expanded], outputs=[x],
                                  name=f'{stage}.slice_x',
                                  starts=[0], ends=[1], axes=[-1]))
        nodes.append(oh.make_node('Slice', inputs=[mesh_grid_expanded], outputs=[y],
                                  name=f'{stage}.slice_y',
                                  starts=[1], ends=[2], axes=[-1]))
        
        one = f'{stage}.one'
        minus_one = f'{stage}.minus_one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        inits.append(oh.make_tensor(minus_one, TensorProto.FLOAT, [], [-1.0]))
        
        x_valid = f'{stage}.x_valid'
        y_valid = f'{stage}.y_valid'
        nodes.append(oh.make_node('And',
                                  inputs=[
                                      oh.make_node('Greater', inputs=[x, minus_one], outputs=['x_gt_minus1']),
                                      oh.make_node('Less', inputs=[x, one], outputs=['x_lt_1'])
                                  ],
                                  outputs=[x_valid],
                                  name=f'{stage}.and_x_valid'))
        nodes.append(oh.make_node('And',
                                  inputs=[
                                      oh.make_node('Greater', inputs=[y, minus_one], outputs=['y_gt_minus1']),
                                      oh.make_node('Less', inputs=[y, one], outputs=['y_lt_1'])
                                  ],
                                  outputs=[y_valid],
                                  name=f'{stage}.and_y_valid'))
        
        valid_mask = f'{stage}.valid_mask'
        nodes.append(oh.make_node('And', inputs=[x_valid, y_valid], outputs=[valid_mask],
                                  name=f'{stage}.and_valid'))
        
        # Unsqueeze for output [1,1,h,w]
        valid_mask_expanded = f'{stage}.valid_mask_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[valid_mask], outputs=[valid_mask_expanded],
                                  name=f'{stage}.unsqueeze_valid_mask', axes=[0]))
        nodes.append(oh.make_node('Unsqueeze', inputs=[valid_mask_expanded], outputs=[valid_mask_expanded],
                                  name=f'{stage}.unsqueeze_valid_mask_2', axes=[0]))
        
        vis.append(oh.make_tensor_value_info(stabilized_frame, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        vis.append(oh.make_tensor_value_info(valid_mask_expanded, TensorProto.BOOL, ['n', 1, 'h', 'w']))
        
        outputs = {
            'stabilized_frame': {'name': stabilized_frame},
            'valid_mask': {'name': valid_mask_expanded}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(current_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(mesh_grid, type=TensorProto.FLOAT, shape=['mesh_h', 'mesh_w', 2])
        result.appendInput(image_size, type=TensorProto.INT64, shape=[2])
        result.appendInput(padding_mode, type=TensorProto.INT64, shape=[1])
        return result

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)