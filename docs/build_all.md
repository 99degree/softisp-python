user@DESKTOP-9JVI26D:~/sources/softisp/onnx$ python build_all.py pipeline.json --mode applier
[DEBUG] Registered ('unnamed', 'v0') → Bayer2CFABase
[DEBUG] Registered ('bayer2cfa', 'v0.bggr') → Bayer2CFA_BGGR
[DEBUG] Registered ('bayer2cfa', 'v0.gbrg') → Bayer2CFA_GBRG
[DEBUG] Registered ('bayer2cfa', 'v0.grbg') → Bayer2CFA_GRBG
[DEBUG] Registered ('bayer2cfa', 'v0.rggb') → Bayer2CFA_RGGB
[DEBUG] Registered ('bayer2float32_base', 'v0') → BayerToFloat
[DEBUG] Registered ('bayernorm_10bit', 'v0') → BayerNorm10
[DEBUG] Registered ('bayernorm_12bit', 'v0') → BayerNorm12
[DEBUG] Registered ('bayernorm_base', 'v0') → BayerNormBase
[DEBUG] Registered ('bayernorm_base', 'v0') → BayerNormBaseV0
[DEBUG] Registered ('blacklevel', 'v0') → BlackLevelBase
[DEBUG] Registered ('blacklevel', 'v0') → BlackLevelBase
[DEBUG] Registered ('blacklevel', 'v2') → BlackLevelV2
[DEBUG] Registered ('ccm_base', 'v0') → CCMBase
[DEBUG] Registered ('ccm_base', 'v0') → CCMBase
[DEBUG] Registered ('ccm_quadratic_v1', 'v1') → CCMQuadraticV1
[DEBUG] Registered ('ccm_quadratic_v1', 'v1') → CCMQuadraticV1
[DEBUG] Registered ('ccm_quadratic_v2', 'v2') → CCMQuadraticV2
[DEBUG] Registered ('chroma_subsample_base', 'v0') → ChromaSubsampleBase
[DEBUG] Registered ('demosaic_avg_lux_v1', 'v1') → DemosaicAvgLuxV1
[DEBUG] Registered ('demosaic_base', 'v0') → DemosaicBase
[DEBUG] Registered ('demosaic_avg_resize_v1', 'v1') → DemosaicAvgResizeV0
[DEBUG] Registered ('demosaic_base', 'v0') → DemosaicBase
[DEBUG] Registered ('demosaic_avg_resize_v1', 'v1') → DemosaicAvgResizeV1
[DEBUG] Registered ('demosaic_base', 'v0') → DemosaicBase
[DEBUG] Registered ('demosaic_avg_v1', 'v1') → DemosaicAvgV1
[DEBUG] Registered ('demosaic_base', 'v0') → DemosaicBase
[DEBUG] Registered ('demosaic_base', 'v0') → DemosaicBase
[DEBUG] Registered ('demosaic_base', 'v0') → DemosaicBase
[DEBUG] Registered ('demosaic_mhc', 'v0') → DemosaicMHC
[DEBUG] Registered ('gamma_base', 'v0') → GammaBase
[DEBUG] Registered ('gamma_base', 'v0') → GammaBase
[DEBUG] Registered ('gamma_v2', 'v2') → GammaV2
[DEBUG] Registered ('image_desc_base', 'v0') → ImageDescBase
[DEBUG] Registered ('image_desc_v1', 'v1-float-float') → ImageDescV1
[DEBUG] Registered ('image_desc_base', 'v0') → ImageDescBase
[DEBUG] Registered ('image_desc_v1', 'v1-int16') → ImageDescV1
[DEBUG] Registered ('lcs_base', 'v0') → LcsBase
[DEBUG] Registered ('lens_lcs_base', 'v0') → LensLCSBase
[DEBUG] Registered ('lens_lcs_base', 'v0') → LensLCSBase
[DEBUG] Registered ('lens_lcs_v1', 'v1') → LensLCSV1
[DEBUG] Registered ('lens_lcs_base', 'v0') → LensLCSBase
[DEBUG] Registered ('lens_lcs_v2', 'v2') → LensLCSV2
[DEBUG] Registered ('resize_base', 'v0') → ResizeBase
[DEBUG] Registered ('resize_base', 'v0') → ResizeBase
[DEBUG] Registered ('resize_v1', 'v1') → ResizeV1
[DEBUG] Registered ('stride_remove_crop', 'v0') → CropWidthFixedBase
[DEBUG] Registered ('tonemap_base', 'v0') → ToneMapBase
[DEBUG] Registered ('tonemap_filmic_v1', 'v1') → ToneFilmicV1
[DEBUG] Registered ('tonemap_base', 'v0') → ToneMapBase
[DEBUG] Registered ('tonemap_filmic_v1', 'v1') → ToneFilmicV1
[DEBUG] Registered ('tonemap_filmic_v2', 'v2') → ToneFilmicV2
[DEBUG] Registered ('tonemap_filmic_v2', 'v2') → ToneFilmicV2
[DEBUG] Registered ('tonemap_filmic_v3', 'v3') → ToneFilmicV3
[DEBUG] Registered ('awb_base', 'v0') → AWBBase
[DEBUG] Registered ('wb_avg_v1', 'v1') → WBAvgV1
[DEBUG] Registered ('awb_base', 'v0') → AWBBase
[DEBUG] Registered ('awb_base', 'v0') → AWBBase
[DEBUG] Registered ('awb_lux_v1', 'v1') → AWBLuxV1
[DEBUG] Registered ('awb_base', 'v0') → AWBBase
[DEBUG] Registered ('awb_base', 'v1') → WhiteBalanceV1
[DEBUG] Registered ('yuvconvert_base', 'v0') → YUVConvertBase
=== Building stages ===

--- Stage image_desc (image_desc_base vv0) ---
[DEBUG] finalize_function: building FunctionProto ImageDescBase_v0
Top-level call node: ['ImageDescBase_v0', 'Constant']
Inputs: ['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function']
Outputs: ['image_desc.applier.function', 'image_desc.width.function', 'image_desc.frame_id.function']

--- Stage bayer2cfa (bayer2cfa vv0.rggb) ---
[DEBUG] finalize_function: building FunctionProto Bayer2CFA_RGGB_v0.rggb
Top-level call node: ['Bayer2CFA_RGGB_v0.rggb', 'Constant']
Inputs: ['image_desc.applier.function']
Outputs: ['bayer2cfa.applier.function', 'bayer2cfa.cfa_onehot.function']

--- Stage bayernorm (bayernorm_10bit vv0) ---
[DEBUG] finalize_function: building FunctionProto BayerNorm10_v0
Top-level call node: ['BayerNorm10_v0', 'Constant']
Inputs: ['bayer2cfa.applier.function']
Outputs: ['bayernorm.applier.function']

--- Stage stride_remove (stride_remove_crop vv0) ---
[DEBUG] finalize_function: building FunctionProto CropWidthFixedBase_v0
Top-level call node: ['CropWidthFixedBase_v0', 'Constant']
Inputs: ['bayernorm.applier.function']
Outputs: ['stride_remove.applier.function']

--- Stage blacklevel (blacklevel vv2) ---
[DEBUG] finalize_function: building FunctionProto BlackLevelV2_v2
Top-level call node: ['BlackLevelV2_v2', 'Constant']
Inputs: ['stride_remove.applier.function', 'blacklevel.offset.function']
Outputs: ['blacklevel.applier.function']

--- Stage demosaic (demosaic_base vv0) ---
[DEBUG] finalize_function: building FunctionProto DemosaicBase_v0
Top-level call node: ['DemosaicBase_v0', 'Constant']
Inputs: ['blacklevel.applier.function']
Outputs: ['demosaic.applier.function']

--- Stage awb (wb_avg_v1 vv1) ---
[DEBUG] finalize_function: building FunctionProto WBAvgV1_v1
Top-level call node: ['WBAvgV1_v1', 'Constant']
Inputs: ['demosaic.applier.function', 'awb.wb_gains.function']
Outputs: ['awb.applier.function']

--- Stage ccm (ccm_quadratic_v1 vv1) ---
[DEBUG] finalize_function: building FunctionProto CCMQuadraticV1_v1
Top-level call node: ['CCMQuadraticV1_v1', 'Constant']
Inputs: ['awb.applier.function', 'ccm.ccm.function']
Outputs: ['ccm.applier.function']

--- Stage tonemap (tonemap_base vv0) ---
[DEBUG] finalize_function: building FunctionProto ToneMapBase_v0
Top-level call node: ['ToneMapBase_v0', 'Constant']
Inputs: ['ccm.applier.function', 'tonemap.tonemap_curve.function']
Outputs: ['tonemap.applier.function']

--- Stage gamma (gamma_base vv0) ---
[DEBUG] finalize_function: building FunctionProto GammaBase_v0
Top-level call node: ['GammaBase_v0', 'Constant']
Inputs: ['tonemap.applier.function', 'gamma.gamma_value.function']
Outputs: ['gamma.applier.function']

--- Stage rgb (image_desc_v1 vv1-int16) ---
gamma_base
image_desc_base
image_desc_base
image_desc_base
[DEBUG] finalize_function: building FunctionProto ImageDescV1_v1-int16
Top-level call node: ['ImageDescV1_v1-int16', 'Constant']
Inputs: ['gamma.applier.function', 'image_desc.frame_id.function']
Outputs: ['rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function']

--- Stage yuv (yuvconvert_base vv0) ---
[DEBUG] finalize_function: building FunctionProto YUVConvertBase_v0
Top-level call node: ['YUVConvertBase_v0', 'Constant']
Inputs: ['gamma.applier.function', 'yuv.rgb2yuv_matrix.function']
Outputs: ['yuv.applier.function']

--- Stage chroma (chroma_subsample_base vv0) ---
[DEBUG] finalize_function: building FunctionProto ChromaSubsampleBase_v0
Top-level call node: ['ChromaSubsampleBase_v0', 'Constant']
Inputs: ['yuv.applier.function', 'chroma.subsample_scale.function']
Outputs: ['chroma.applier.function']
DEBUG:[DIAG] func ImageDescBase_v0 value_info: ['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'image_desc.applier.function', 'image_desc.width.function', 'image_desc.frame_id.function']
DEBUG:[DIAG] func Bayer2CFA_RGGB_v0.rggb value_info: ['image_desc.applier.function', 'bayer2cfa.applier.function', 'bayer2cfa.cfa_onehot.function']
DEBUG:[DIAG] func BayerNorm10_v0 value_info: ['bayer2cfa.applier.function', 'bayernorm.applier.function']
DEBUG:[DIAG] func CropWidthFixedBase_v0 value_info: ['bayernorm.applier.function', 'stride_remove.applier.function']
DEBUG:[DIAG] func BlackLevelV2_v2 value_info: ['stride_remove.applier.function', 'blacklevel.offset.function', 'blacklevel.applier.function']
DEBUG:[DIAG] func DemosaicBase_v0 value_info: ['blacklevel.applier.function', 'demosaic.applier.function']
DEBUG:[DIAG] func WBAvgV1_v1 value_info: ['demosaic.applier.function', 'awb.wb_gains.function', 'awb.applier.function']
DEBUG:[DIAG] func CCMQuadraticV1_v1 value_info: ['awb.applier.function', 'ccm.ccm.function', 'ccm.applier.function']
DEBUG:[DIAG] func ToneMapBase_v0 value_info: ['ccm.applier.function', 'tonemap.tonemap_curve.function', 'tonemap.applier.function']
DEBUG:[DIAG] func GammaBase_v0 value_info: ['tonemap.applier.function', 'gamma.gamma_value.function', 'gamma.applier.function']
DEBUG:[DIAG] func ImageDescV1_v1-int16 value_info: ['gamma.applier.function', 'image_desc.frame_id.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function']
DEBUG:[DIAG] func YUVConvertBase_v0 value_info: ['gamma.applier.function', 'yuv.rgb2yuv_matrix.function', 'yuv.applier.function']
DEBUG:[DIAG] func ChromaSubsampleBase_v0 value_info: ['yuv.applier.function', 'chroma.subsample_scale.function', 'chroma.applier.function']
DEBUG:[DIAG] result.inputs[image_desc.input.image.function] keys=['name']
DEBUG:[DIAG] result.inputs[image_desc.input.width.function] keys=['name']
DEBUG:[DIAG] result.inputs[image_desc.input.frame_id.function] keys=['name']
DEBUG:[DIAG] result.inputs[image_desc.applier.function] keys=['name']
DEBUG:[DIAG] result.inputs[bayer2cfa.applier.function] keys=['name']
DEBUG:[DIAG] result.inputs[bayernorm.applier.function] keys=['name']
DEBUG:[DIAG] result.inputs[stride_remove.applier.function] keys=['name']
DEBUG:[DIAG] result.inputs[blacklevel.offset.function] keys=['name']
DEBUG:[DIAG] result.inputs[blacklevel.applier.function] keys=['name']
DEBUG:[DIAG] result.inputs[demosaic.applier.function] keys=['name']
DEBUG:[DIAG] result.inputs[awb.wb_gains.function] keys=['name']
DEBUG:[DIAG] result.inputs[awb.applier.function] keys=['name']
DEBUG:[DIAG] result.inputs[ccm.ccm.function] keys=['name']
DEBUG:[DIAG] result.inputs[ccm.applier.function] keys=['name']
DEBUG:[DIAG] result.inputs[tonemap.tonemap_curve.function] keys=['name']
DEBUG:[DIAG] result.inputs[tonemap.applier.function] keys=['name']
DEBUG:[DIAG] result.inputs[gamma.gamma_value.function] keys=['name']
DEBUG:[DIAG] result.inputs[gamma.applier.function] keys=['name']
DEBUG:[DIAG] result.inputs[image_desc.frame_id.function] keys=['name']
DEBUG:[DIAG] result.inputs[gamma.applier.function] keys=['name']
DEBUG:[DIAG] result.inputs[yuv.rgb2yuv_matrix.function] keys=['name']
DEBUG:[DIAG] result.inputs[yuv.applier.function] keys=['name']
DEBUG:[DIAG] result.inputs[chroma.subsample_scale.function] keys=['name']
DEBUG:[DIAG] vis names before promotion: ['ImageDescBase.image_desc_base.v0_marker', 'Bayer2CFA_RGGB.bayer2cfa.v0.rggb_marker', 'BayerNorm10.bayernorm_10bit.v0_marker', 'CropWidthFixedBase.stride_remove_crop.v0_marker', 'BlackLevelV2.blacklevel.v2_marker', 'DemosaicBase.demosaic_base.v0_marker', 'WBAvgV1.wb_avg_v1.v1_marker', 'CCMQuadraticV1.ccm_quadratic_v1.v1_marker', 'ToneMapBase.tonemap_base.v0_marker', 'GammaBase.gamma_base.v0_marker', 'ImageDescV1.image_desc_v1.v1-int16_marker', 'YUVConvertBase.yuvconvert_base.v0_marker', 'ChromaSubsampleBase.chroma_subsample_base.v0_marker']
DEBUG:[PROMOTE] Promoted input 'image_desc.input.image.function' using existing ValueInfo type=5 shape=['h', 'w']
DEBUG:[PROMOTE] Promoted input 'image_desc.input.width.function' using existing ValueInfo type=7 shape=[1]
DEBUG:[PROMOTE] Promoted input 'image_desc.input.frame_id.function' using existing ValueInfo type=7 shape=[1]
DEBUG:[PROMOTE] Promoted input 'blacklevel.offset.function' using existing ValueInfo type=1 shape=[1]
DEBUG:[PROMOTE] Promoted input 'awb.wb_gains.function' using existing ValueInfo type=1 shape=[1, 3, 1, 1]
DEBUG:[PROMOTE] Promoted input 'ccm.ccm.function' using existing ValueInfo type=1 shape=[3, 3, 1, 1]
DEBUG:[PROMOTE] Promoted input 'tonemap.tonemap_curve.function' using existing ValueInfo type=1 shape=[1]
DEBUG:[PROMOTE] Promoted input 'gamma.gamma_value.function' using existing ValueInfo type=1 shape=[1]
DEBUG:[PROMOTE] Promoted input 'yuv.rgb2yuv_matrix.function' using existing ValueInfo type=1 shape=[3, 3, 1, 1]
DEBUG:[PROMOTE] Promoted input 'chroma.subsample_scale.function' using existing ValueInfo type=1 shape=[4]
DEBUG:[DEBUG] Consumed inputs: ['awb.applier.function', 'tonemap.tonemap_curve.function', 'yuv.rgb2yuv_matrix.function', 'ccm.applier.function', 'yuv.applier.function', 'demosaic.applier.function', 'bayernorm.applier.function', 'awb.wb_gains.function', 'tonemap.applier.function', 'image_desc.applier.function', 'image_desc.frame_id.function', 'stride_remove.applier.function', 'image_desc.input.width.function', 'gamma.gamma_value.function', 'blacklevel.offset.function', 'blacklevel.applier.function', 'image_desc.input.image.function', 'bayer2cfa.applier.function', 'chroma.subsample_scale.function', 'image_desc.input.frame_id.function', 'gamma.applier.function', 'ccm.ccm.function']

=== Prod mode: promoting dangling outputs only ===
DEBUG:[SKIP] image_desc.applier.function skipped (consumed downstream)
DEBUG:[SKIP] image_desc.width.function skipped (already node output)
DEBUG:[SKIP] image_desc.frame_id.function skipped (consumed downstream)
DEBUG:[SKIP] bayer2cfa.applier.function skipped (consumed downstream)
DEBUG:[SKIP] bayer2cfa.cfa_onehot.function skipped (already node output)
DEBUG:[SKIP] bayernorm.applier.function skipped (consumed downstream)
DEBUG:[SKIP] stride_remove.applier.function skipped (consumed downstream)
DEBUG:[SKIP] blacklevel.applier.function skipped (consumed downstream)
DEBUG:[SKIP] demosaic.applier.function skipped (consumed downstream)
DEBUG:[SKIP] awb.applier.function skipped (consumed downstream)
DEBUG:[SKIP] ccm.applier.function skipped (consumed downstream)
DEBUG:[SKIP] tonemap.applier.function skipped (consumed downstream)
DEBUG:[SKIP] gamma.applier.function skipped (consumed downstream)
DEBUG:[SKIP] rgb.rgb_out.function skipped (already node output)
DEBUG:[SKIP] rgb.height.function skipped (already node output)
DEBUG:[SKIP] rgb.width.function skipped (already node output)
DEBUG:[SKIP] rgb.frame_id.function skipped (already node output)
DEBUG:[SKIP] yuv.applier.function skipped (consumed downstream)
DEBUG:[SKIP] chroma.applier.function skipped (already node output)
WARNING:[PROMOTE] No promoted outputs found; promoting dangling stage outputs as graph outputs
DEBUG:[PROMOTE-BRANCH] Promoted dangling stage output (function): image_desc.width.function
DEBUG:[PROMOTE-BRANCH] Promoted dangling stage output (function): bayer2cfa.cfa_onehot.function
DEBUG:[PROMOTE-BRANCH] Promoted dangling stage output (function): rgb.rgb_out.function
DEBUG:[PROMOTE-BRANCH] Promoted dangling stage output (function): rgb.height.function
DEBUG:[PROMOTE-BRANCH] Promoted dangling stage output (function): rgb.width.function
DEBUG:[PROMOTE-BRANCH] Promoted dangling stage output (function): rgb.frame_id.function
DEBUG:[PROMOTE-BRANCH] Promoted dangling stage output (function): chroma.applier.function

Promoted outputs: ['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'chroma.applier.function']
DEBUG:[FILTER] value_info after cleanup: []
DEBUG:[SANITIZE] Checking function protos for name collisions...
DEBUG:[FUNC SANITIZE] Renaming function output 'image_desc.applier.function' -> 'ImageDescBase_v0.image_desc.applier.function' in function ImageDescBase_v0
DEBUG:[FUNC SANITIZE] Renaming function output 'image_desc.width.function' -> 'ImageDescBase_v0.image_desc.width.function' in function ImageDescBase_v0
DEBUG:[FUNC SANITIZE] Renaming function output 'image_desc.frame_id.function' -> 'ImageDescBase_v0.image_desc.frame_id.function' in function ImageDescBase_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'image_desc.applier.function' -> 'ImageDescBase_v0.image_desc.applier.function' in function ImageDescBase_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'image_desc.width.function' -> 'ImageDescBase_v0.image_desc.width.function' in function ImageDescBase_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'image_desc.frame_id.function' -> 'ImageDescBase_v0.image_desc.frame_id.function' in function ImageDescBase_v0
DEBUG:[FUNC SANITIZE] Renaming node output 'image_desc.applier.function' -> 'ImageDescBase_v0.image_desc.applier.function' inside function ImageDescBase_v0
DEBUG:[FUNC SANITIZE] Renaming node output 'image_desc.width.function' -> 'ImageDescBase_v0.image_desc.width.function' inside function ImageDescBase_v0
DEBUG:[FUNC SANITIZE] Renaming node output 'image_desc.frame_id.function' -> 'ImageDescBase_v0.image_desc.frame_id.function' inside function ImageDescBase_v0
DEBUG:[FUNC SANITIZE] Renaming function output 'bayer2cfa.applier.function' -> 'Bayer2CFA_RGGB_v0.rggb.bayer2cfa.applier.function' in function Bayer2CFA_RGGB_v0.rggb
DEBUG:[FUNC SANITIZE] Renaming function output 'bayer2cfa.cfa_onehot.function' -> 'Bayer2CFA_RGGB_v0.rggb.bayer2cfa.cfa_onehot.function' in function Bayer2CFA_RGGB_v0.rggb
DEBUG:[FUNC SANITIZE] Renaming function value_info 'image_desc.applier.function' -> 'Bayer2CFA_RGGB_v0.rggb.image_desc.applier.function' in function Bayer2CFA_RGGB_v0.rggb
DEBUG:[FUNC SANITIZE] Renaming function value_info 'bayer2cfa.applier.function' -> 'Bayer2CFA_RGGB_v0.rggb.bayer2cfa.applier.function' in function Bayer2CFA_RGGB_v0.rggb
DEBUG:[FUNC SANITIZE] Renaming function value_info 'bayer2cfa.cfa_onehot.function' -> 'Bayer2CFA_RGGB_v0.rggb.bayer2cfa.cfa_onehot.function' in function Bayer2CFA_RGGB_v0.rggb
DEBUG:[FUNC SANITIZE] Renaming node output 'bayer2cfa.applier.function' -> 'Bayer2CFA_RGGB_v0.rggb.bayer2cfa.applier.function' inside function Bayer2CFA_RGGB_v0.rggb
DEBUG:[FUNC SANITIZE] Renaming node output 'bayer2cfa.cfa_onehot.function' -> 'Bayer2CFA_RGGB_v0.rggb.bayer2cfa.cfa_onehot.function' inside function Bayer2CFA_RGGB_v0.rggb
DEBUG:[FUNC SANITIZE] Renaming function output 'bayernorm.applier.function' -> 'BayerNorm10_v0.bayernorm.applier.function' in function BayerNorm10_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'bayer2cfa.applier.function' -> 'BayerNorm10_v0.bayer2cfa.applier.function' in function BayerNorm10_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'bayernorm.applier.function' -> 'BayerNorm10_v0.bayernorm.applier.function' in function BayerNorm10_v0
DEBUG:[FUNC SANITIZE] Renaming node output 'bayernorm.applier.function' -> 'BayerNorm10_v0.bayernorm.applier.function' inside function BayerNorm10_v0
DEBUG:[FUNC SANITIZE] Renaming function output 'stride_remove.applier.function' -> 'CropWidthFixedBase_v0.stride_remove.applier.function' in function CropWidthFixedBase_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'bayernorm.applier.function' -> 'CropWidthFixedBase_v0.bayernorm.applier.function' in function CropWidthFixedBase_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'stride_remove.applier.function' -> 'CropWidthFixedBase_v0.stride_remove.applier.function' in function CropWidthFixedBase_v0
DEBUG:[FUNC SANITIZE] Renaming node output 'stride_remove.applier.function' -> 'CropWidthFixedBase_v0.stride_remove.applier.function' inside function CropWidthFixedBase_v0
DEBUG:[FUNC SANITIZE] Renaming function output 'blacklevel.applier.function' -> 'BlackLevelV2_v2.blacklevel.applier.function' in function BlackLevelV2_v2
DEBUG:[FUNC SANITIZE] Renaming function value_info 'stride_remove.applier.function' -> 'BlackLevelV2_v2.stride_remove.applier.function' in function BlackLevelV2_v2
DEBUG:[FUNC SANITIZE] Renaming function value_info 'blacklevel.applier.function' -> 'BlackLevelV2_v2.blacklevel.applier.function' in function BlackLevelV2_v2
DEBUG:[FUNC SANITIZE] Renaming node output 'blacklevel.applier.function' -> 'BlackLevelV2_v2.blacklevel.applier.function' inside function BlackLevelV2_v2
DEBUG:[FUNC SANITIZE] Renaming function output 'demosaic.applier.function' -> 'DemosaicBase_v0.demosaic.applier.function' in function DemosaicBase_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'blacklevel.applier.function' -> 'DemosaicBase_v0.blacklevel.applier.function' in function DemosaicBase_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'demosaic.applier.function' -> 'DemosaicBase_v0.demosaic.applier.function' in function DemosaicBase_v0
DEBUG:[FUNC SANITIZE] Renaming node output 'demosaic.applier.function' -> 'DemosaicBase_v0.demosaic.applier.function' inside function DemosaicBase_v0
DEBUG:[FUNC SANITIZE] Renaming function output 'awb.applier.function' -> 'WBAvgV1_v1.awb.applier.function' in function WBAvgV1_v1
DEBUG:[FUNC SANITIZE] Renaming function value_info 'demosaic.applier.function' -> 'WBAvgV1_v1.demosaic.applier.function' in function WBAvgV1_v1
DEBUG:[FUNC SANITIZE] Renaming function value_info 'awb.applier.function' -> 'WBAvgV1_v1.awb.applier.function' in function WBAvgV1_v1
DEBUG:[FUNC SANITIZE] Renaming node output 'awb.applier.function' -> 'WBAvgV1_v1.awb.applier.function' inside function WBAvgV1_v1
DEBUG:[FUNC SANITIZE] Renaming function output 'ccm.applier.function' -> 'CCMQuadraticV1_v1.ccm.applier.function' in function CCMQuadraticV1_v1
DEBUG:[FUNC SANITIZE] Renaming function value_info 'awb.applier.function' -> 'CCMQuadraticV1_v1.awb.applier.function' in function CCMQuadraticV1_v1
DEBUG:[FUNC SANITIZE] Renaming function value_info 'ccm.applier.function' -> 'CCMQuadraticV1_v1.ccm.applier.function' in function CCMQuadraticV1_v1
DEBUG:[FUNC SANITIZE] Renaming node output 'ccm.applier.function' -> 'CCMQuadraticV1_v1.ccm.applier.function' inside function CCMQuadraticV1_v1
DEBUG:[FUNC SANITIZE] Renaming function output 'tonemap.applier.function' -> 'ToneMapBase_v0.tonemap.applier.function' in function ToneMapBase_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'ccm.applier.function' -> 'ToneMapBase_v0.ccm.applier.function' in function ToneMapBase_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'tonemap.applier.function' -> 'ToneMapBase_v0.tonemap.applier.function' in function ToneMapBase_v0
DEBUG:[FUNC SANITIZE] Renaming node output 'tonemap.applier.function' -> 'ToneMapBase_v0.tonemap.applier.function' inside function ToneMapBase_v0
DEBUG:[FUNC SANITIZE] Renaming function output 'gamma.applier.function' -> 'GammaBase_v0.gamma.applier.function' in function GammaBase_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'tonemap.applier.function' -> 'GammaBase_v0.tonemap.applier.function' in function GammaBase_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'gamma.applier.function' -> 'GammaBase_v0.gamma.applier.function' in function GammaBase_v0
DEBUG:[FUNC SANITIZE] Renaming node output 'gamma.applier.function' -> 'GammaBase_v0.gamma.applier.function' inside function GammaBase_v0
DEBUG:[FUNC SANITIZE] Renaming function output 'rgb.rgb_out.function' -> 'ImageDescV1_v1-int16.rgb.rgb_out.function' in function ImageDescV1_v1-int16
DEBUG:[FUNC SANITIZE] Renaming function output 'rgb.height.function' -> 'ImageDescV1_v1-int16.rgb.height.function' in function ImageDescV1_v1-int16
DEBUG:[FUNC SANITIZE] Renaming function output 'rgb.width.function' -> 'ImageDescV1_v1-int16.rgb.width.function' in function ImageDescV1_v1-int16
DEBUG:[FUNC SANITIZE] Renaming function output 'rgb.frame_id.function' -> 'ImageDescV1_v1-int16.rgb.frame_id.function' in function ImageDescV1_v1-int16
DEBUG:[FUNC SANITIZE] Renaming function value_info 'gamma.applier.function' -> 'ImageDescV1_v1-int16.gamma.applier.function' in function ImageDescV1_v1-int16
DEBUG:[FUNC SANITIZE] Renaming function value_info 'image_desc.frame_id.function' -> 'ImageDescV1_v1-int16.image_desc.frame_id.function' in function ImageDescV1_v1-int16
DEBUG:[FUNC SANITIZE] Renaming function value_info 'rgb.rgb_out.function' -> 'ImageDescV1_v1-int16.rgb.rgb_out.function' in function ImageDescV1_v1-int16
DEBUG:[FUNC SANITIZE] Renaming function value_info 'rgb.height.function' -> 'ImageDescV1_v1-int16.rgb.height.function' in function ImageDescV1_v1-int16
DEBUG:[FUNC SANITIZE] Renaming function value_info 'rgb.width.function' -> 'ImageDescV1_v1-int16.rgb.width.function' in function ImageDescV1_v1-int16
DEBUG:[FUNC SANITIZE] Renaming function value_info 'rgb.frame_id.function' -> 'ImageDescV1_v1-int16.rgb.frame_id.function' in function ImageDescV1_v1-int16
DEBUG:[FUNC SANITIZE] Renaming node output 'image_desc.frame_id.function' -> 'ImageDescV1_v1-int16.image_desc.frame_id.function' inside function ImageDescV1_v1-int16
DEBUG:[FUNC SANITIZE] Renaming node output 'rgb.rgb_out.function' -> 'ImageDescV1_v1-int16.rgb.rgb_out.function' inside function ImageDescV1_v1-int16
DEBUG:[FUNC SANITIZE] Renaming node output 'rgb.height.function' -> 'ImageDescV1_v1-int16.rgb.height.function' inside function ImageDescV1_v1-int16
DEBUG:[FUNC SANITIZE] Renaming node output 'rgb.width.function' -> 'ImageDescV1_v1-int16.rgb.width.function' inside function ImageDescV1_v1-int16
DEBUG:[FUNC SANITIZE] Renaming node output 'rgb.frame_id.function' -> 'ImageDescV1_v1-int16.rgb.frame_id.function' inside function ImageDescV1_v1-int16
DEBUG:[FUNC SANITIZE] Renaming function output 'yuv.applier.function' -> 'YUVConvertBase_v0.yuv.applier.function' in function YUVConvertBase_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'gamma.applier.function' -> 'YUVConvertBase_v0.gamma.applier.function' in function YUVConvertBase_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'yuv.applier.function' -> 'YUVConvertBase_v0.yuv.applier.function' in function YUVConvertBase_v0
DEBUG:[FUNC SANITIZE] Renaming node output 'yuv.applier.function' -> 'YUVConvertBase_v0.yuv.applier.function' inside function YUVConvertBase_v0
DEBUG:[FUNC SANITIZE] Renaming function output 'chroma.applier.function' -> 'ChromaSubsampleBase_v0.chroma.applier.function' in function ChromaSubsampleBase_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'yuv.applier.function' -> 'ChromaSubsampleBase_v0.yuv.applier.function' in function ChromaSubsampleBase_v0
DEBUG:[FUNC SANITIZE] Renaming function value_info 'chroma.applier.function' -> 'ChromaSubsampleBase_v0.chroma.applier.function' in function ChromaSubsampleBase_v0
DEBUG:[FUNC SANITIZE] Renaming node output 'chroma.applier.function' -> 'ChromaSubsampleBase_v0.chroma.applier.function' inside function ChromaSubsampleBase_v0
DEBUG:Function attached: softisp:ImageDescBase_v0
DEBUG:Function attached: softisp:Bayer2CFA_RGGB_v0.rggb
DEBUG:Function attached: softisp:BayerNorm10_v0
DEBUG:Function attached: softisp:CropWidthFixedBase_v0
DEBUG:Function attached: softisp:BlackLevelV2_v2
DEBUG:Function attached: softisp:DemosaicBase_v0
DEBUG:Function attached: softisp:WBAvgV1_v1
DEBUG:Function attached: softisp:CCMQuadraticV1_v1
DEBUG:Function attached: softisp:ToneMapBase_v0
DEBUG:Function attached: softisp:GammaBase_v0
DEBUG:Function attached: softisp:ImageDescV1_v1-int16
DEBUG:Function attached: softisp:YUVConvertBase_v0
DEBUG:Function attached: softisp:ChromaSubsampleBase_v0
DEBUG:=== Graph Outputs BEFORE model ===
DEBUG:Output: image_desc.width.function
DEBUG:Output: bayer2cfa.cfa_onehot.function
DEBUG:Output: rgb.rgb_out.function
DEBUG:Output: rgb.height.function
DEBUG:Output: rgb.width.function
DEBUG:Output: rgb.frame_id.function
DEBUG:Output: chroma.applier.function
DEBUG:=== Final Graph Outputs ===
DEBUG:Output: image_desc.width.function
DEBUG:Output: bayer2cfa.cfa_onehot.function
DEBUG:Output: rgb.rgb_out.function
DEBUG:Output: rgb.height.function
DEBUG:Output: rgb.width.function
DEBUG:Output: rgb.frame_id.function
DEBUG:Output: chroma.applier.function
DEBUG:Graph node output: image_desc.applier.function  type=tensor(float)  shape=[1, 1, 'h', 'w']
DEBUG:Graph node output: image_desc.width.function  type=tensor(int64)  shape=[1]
DEBUG:Graph node output: image_desc.frame_id.function  type=tensor(int64)  shape=[1]
DEBUG:Graph node output: ImageDescBase.image_desc_base.v0_marker  type=UNKNOWN  shape=UNKNOWN
DEBUG:Graph node output: bayer2cfa.applier.function  type=tensor(float)  shape=['n', 4, 'h', 'w']
DEBUG:Graph node output: bayer2cfa.cfa_onehot.function  type=tensor(int64)  shape=['n', 2, 2, 4]
DEBUG:Graph node output: Bayer2CFA_RGGB.bayer2cfa.v0.rggb_marker  type=UNKNOWN  shape=UNKNOWN
DEBUG:Graph node output: bayernorm.applier.function  type=tensor(float)  shape=['n', 4, 'h', 'w']
DEBUG:Graph node output: BayerNorm10.bayernorm_10bit.v0_marker  type=UNKNOWN  shape=UNKNOWN
DEBUG:Graph node output: stride_remove.applier.function  type=tensor(float)  shape=['n', 4, 'H', 'W']
DEBUG:Graph node output: CropWidthFixedBase.stride_remove_crop.v0_marker  type=UNKNOWN  shape=UNKNOWN
DEBUG:Graph node output: blacklevel.applier.function  type=tensor(float)  shape=['N', 4, 'H', 'W']
DEBUG:Graph node output: BlackLevelV2.blacklevel.v2_marker  type=UNKNOWN  shape=UNKNOWN
DEBUG:Graph node output: demosaic.applier.function  type=tensor(float)  shape=['n', 3, 'h', 'w']
DEBUG:Graph node output: DemosaicBase.demosaic_base.v0_marker  type=UNKNOWN  shape=UNKNOWN
DEBUG:Graph node output: awb.applier.function  type=tensor(float)  shape=[1, 3, 'h', 'w']
DEBUG:Graph node output: WBAvgV1.wb_avg_v1.v1_marker  type=UNKNOWN  shape=UNKNOWN
DEBUG:Graph node output: ccm.applier.function  type=tensor(float)  shape=['n', 3, 'h', 'w']
DEBUG:Graph node output: CCMQuadraticV1.ccm_quadratic_v1.v1_marker  type=UNKNOWN  shape=UNKNOWN
DEBUG:Graph node output: tonemap.applier.function  type=tensor(float)  shape=['n', 3, 'h', 'w']
DEBUG:Graph node output: ToneMapBase.tonemap_base.v0_marker  type=UNKNOWN  shape=UNKNOWN
DEBUG:Graph node output: gamma.applier.function  type=tensor(float)  shape=['n', 3, 'h', 'w']
DEBUG:Graph node output: GammaBase.gamma_base.v0_marker  type=UNKNOWN  shape=UNKNOWN
DEBUG:Graph node output: rgb.rgb_out.function  type=tensor(int16)  shape=['h', 'w', 3]
DEBUG:Graph node output: rgb.height.function  type=tensor(int64)  shape=[1]
DEBUG:Graph node output: rgb.width.function  type=tensor(int64)  shape=[1]
DEBUG:Graph node output: rgb.frame_id.function  type=tensor(int64)  shape=[1]
DEBUG:Graph node output: ImageDescV1.image_desc_v1.v1-int16_marker  type=UNKNOWN  shape=UNKNOWN
DEBUG:Graph node output: yuv.applier.function  type=tensor(float)  shape=['n', 3, 'h', 'w']
DEBUG:Graph node output: YUVConvertBase.yuvconvert_base.v0_marker  type=UNKNOWN  shape=UNKNOWN
DEBUG:Graph node output: chroma.applier.function  type=tensor(float)  shape=['n', 3, 'h', 'w']
DEBUG:Graph node output: ChromaSubsampleBase.chroma_subsample_base.v0_marker  type=UNKNOWN  shape=UNKNOWN
[SSA ERROR] Duplicate name 'image_desc.width.function' found in graph_output, already seen in node:ImageDescBase_v0
[SSA ERROR] Duplicate name 'bayer2cfa.cfa_onehot.function' found in graph_output, already seen in node:Bayer2CFA_RGGB_v0.rggb
[SSA ERROR] Duplicate name 'rgb.rgb_out.function' found in graph_output, already seen in node:ImageDescV1_v1-int16
[SSA ERROR] Duplicate name 'rgb.height.function' found in graph_output, already seen in node:ImageDescV1_v1-int16
[SSA ERROR] Duplicate name 'rgb.width.function' found in graph_output, already seen in node:ImageDescV1_v1-int16
[SSA ERROR] Duplicate name 'rgb.frame_id.function' found in graph_output, already seen in node:ImageDescV1_v1-int16
[SSA ERROR] Duplicate name 'chroma.applier.function' found in graph_output, already seen in node:ChromaSubsampleBase_v0
INFO:Saved ONNX model to onnx_out/softisp_pipeline_test/applier.onnx
