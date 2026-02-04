user@DESKTOP-9JVI26D:~/sources/softisp/onnx$ python test_full_pipeline.py --model_dir onnx_out/softisp_pipeline_test
2026-02-04 23:22:02,106 [INFO] MainThread [Main] parse args
2026-02-04 23:22:02,119 [INFO] MainThread [Main] load test_algo: onnx_out/softisp_pipeline_test/test_algo.onnx
2026-02-04 23:22:02,316 [INFO] MainThread === test_algo.onnx inputs ===
2026-02-04 23:22:02,317 [INFO] MainThread === test_algo.onnx outputs ===
2026-02-04 23:22:02,317 [INFO] MainThread Output 0: name=rgb.input.image, type=tensor(int16), shape=[1080, 2048]
2026-02-04 23:22:02,317 [INFO] MainThread Output 1: name=rgb.input.width, type=tensor(int64), shape=[1]
2026-02-04 23:22:02,317 [INFO] MainThread Output 2: name=rgb.input.frame_id, type=tensor(int64), shape=[1]
2026-02-04 23:22:02,317 [INFO] MainThread [Main] load algo: onnx_out/softisp_pipeline_test/algo.onnx
2026-02-04 23:22:02.321705385 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'ChromaSubsampleBase.chroma_subsample_base.v0_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.321812784 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'YUVConvertBase.yuvconvert_base.v0_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.321822784 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'ImageDescV1.image_desc_v1.v1-int16_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.321941184 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'GammaBase.gamma_base.v0_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.321960284 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'CCMQuadraticV1.ccm_quadratic_v1.v1_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.322066783 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'WBAvgV1.wb_avg_v1.v1_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.322407680 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'DemosaicBase.demosaic_base.v0_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.322529280 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'ToneMapBase.tonemap_base.v0_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.322543880 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'BlackLevelV2.blacklevel.v2_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.322596579 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'CropWidthFixedBase.stride_remove_crop.v0_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.322641279 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'Bayer2CFA_RGGB.bayer2cfa.v0.rggb_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.322820378 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'BayerNorm10.bayernorm_10bit.v0_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.322832178 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'ImageDescBase.image_desc_base.v0_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02,332 [INFO] MainThread === algo.onnx inputs ===
2026-02-04 23:22:02,332 [INFO] MainThread Input 0: name=image_desc.input.image.function, type=tensor(int16), shape=['H', 'W']
2026-02-04 23:22:02,332 [INFO] MainThread Input 1: name=image_desc.input.width.function, type=tensor(int64), shape=[1]
2026-02-04 23:22:02,332 [INFO] MainThread Input 2: name=image_desc.input.frame_id.function, type=tensor(int64), shape=[1]
2026-02-04 23:22:02,332 [INFO] MainThread Input 3: name=blacklevel.offset.function, type=tensor(float), shape=[1]
2026-02-04 23:22:02,333 [INFO] MainThread === algo.onnx outputs ===
2026-02-04 23:22:02,333 [INFO] MainThread Output 0: name=image_desc.width.function, type=tensor(int64), shape=[1]
2026-02-04 23:22:02,333 [INFO] MainThread Output 1: name=bayer2cfa.cfa_onehot.function, type=tensor(int64), shape=[1, 2, 2, 4]
2026-02-04 23:22:02,333 [INFO] MainThread Output 2: name=awb.wb_gains.function, type=tensor(float), shape=[1, 3, 1, 1]
2026-02-04 23:22:02,333 [INFO] MainThread Output 3: name=ccm.ccm.normalized.function, type=tensor(float), shape=[3, 3]
2026-02-04 23:22:02,333 [INFO] MainThread Output 4: name=ccm.ccm.function, type=tensor(float), shape=[3, 3, 1, 1]
2026-02-04 23:22:02,334 [INFO] MainThread Output 5: name=tonemap.tonemap_curve.function, type=tensor(float), shape=[1]
2026-02-04 23:22:02,334 [INFO] MainThread Output 6: name=gamma.gamma_value.function, type=tensor(float), shape=[1]
2026-02-04 23:22:02,334 [INFO] MainThread Output 7: name=rgb.rgb_out.function, type=tensor(int16), shape=['H', 'W', 3]
2026-02-04 23:22:02,334 [INFO] MainThread Output 8: name=rgb.height.function, type=tensor(int64), shape=[1]
2026-02-04 23:22:02,334 [INFO] MainThread Output 9: name=rgb.width.function, type=tensor(int64), shape=[1]
2026-02-04 23:22:02,335 [INFO] MainThread Output 10: name=rgb.frame_id.function, type=tensor(int64), shape=[1]
2026-02-04 23:22:02,335 [INFO] MainThread Output 11: name=yuv.rgb2yuv_matrix.normalized.function, type=tensor(float), shape=[3, 3]
2026-02-04 23:22:02,335 [INFO] MainThread Output 12: name=yuv.rgb2yuv_matrix.function, type=tensor(float), shape=[3, 3, 1, 1]
2026-02-04 23:22:02,335 [INFO] MainThread Output 13: name=chroma.applier.function, type=tensor(float), shape=[1, 3, 'H', 'W']
2026-02-04 23:22:02,335 [INFO] MainThread Output 14: name=chroma.subsample_scale.function, type=tensor(float), shape=[4]
2026-02-04 23:22:02,336 [INFO] MainThread [Main] load ISP: onnx_out/softisp_pipeline_test/applier.onnx
2026-02-04 23:22:02.339961962 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'ChromaSubsampleBase.chroma_subsample_base.v0_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.340036662 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'YUVConvertBase.yuvconvert_base.v0_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.340053761 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'ImageDescV1.image_desc_v1.v1-int16_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.340098161 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'GammaBase.gamma_base.v0_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.340115461 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'CCMQuadraticV1.ccm_quadratic_v1.v1_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.340160461 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'WBAvgV1.wb_avg_v1.v1_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.340183861 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'DemosaicBase.demosaic_base.v0_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.340291060 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'ToneMapBase.tonemap_base.v0_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.340521858 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'BlackLevelV2.blacklevel.v2_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.340575958 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'CropWidthFixedBase.stride_remove_crop.v0_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.340627858 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'Bayer2CFA_RGGB.bayer2cfa.v0.rggb_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.340676157 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'BayerNorm10.bayernorm_10bit.v0_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02.340724857 [W:onnxruntime:, graph.cc:4859 CleanUnusedInitializersAndNodeArgs] Removing initializer 'ImageDescBase.image_desc_base.v0_marker'. It is not used by any node and should be removed from the model.
2026-02-04 23:22:02,347 [INFO] MainThread === applier.onnx inputs ===
2026-02-04 23:22:02,347 [INFO] MainThread Input 0: name=image_desc.input.image.function, type=tensor(int16), shape=['H', 'W']
2026-02-04 23:22:02,347 [INFO] MainThread Input 1: name=image_desc.input.width.function, type=tensor(int64), shape=[1]
2026-02-04 23:22:02,347 [INFO] MainThread Input 2: name=image_desc.input.frame_id.function, type=tensor(int64), shape=[1]
2026-02-04 23:22:02,347 [INFO] MainThread Input 3: name=blacklevel.offset.function, type=tensor(float), shape=[1]
2026-02-04 23:22:02,347 [INFO] MainThread Input 4: name=awb.wb_gains.function, type=tensor(float), shape=[1, 3, 1, 1]
2026-02-04 23:22:02,347 [INFO] MainThread Input 5: name=ccm.ccm.function, type=tensor(float), shape=[3, 3, 1, 1]
2026-02-04 23:22:02,347 [INFO] MainThread Input 6: name=tonemap.tonemap_curve.function, type=tensor(float), shape=[1]
2026-02-04 23:22:02,347 [INFO] MainThread Input 7: name=gamma.gamma_value.function, type=tensor(float), shape=[1]
2026-02-04 23:22:02,347 [INFO] MainThread Input 8: name=yuv.rgb2yuv_matrix.function, type=tensor(float), shape=[3, 3, 1, 1]
2026-02-04 23:22:02,347 [INFO] MainThread Input 9: name=chroma.subsample_scale.function, type=tensor(float), shape=[4]
2026-02-04 23:22:02,348 [INFO] MainThread === applier.onnx outputs ===
2026-02-04 23:22:02,348 [INFO] MainThread Output 0: name=image_desc.width.function, type=tensor(int64), shape=[1]
2026-02-04 23:22:02,348 [INFO] MainThread Output 1: name=bayer2cfa.cfa_onehot.function, type=tensor(int64), shape=[1, 2, 2, 4]
2026-02-04 23:22:02,348 [INFO] MainThread Output 2: name=rgb.rgb_out.function, type=tensor(int16), shape=['H', 'W', 3]
2026-02-04 23:22:02,348 [INFO] MainThread Output 3: name=rgb.height.function, type=tensor(int64), shape=[1]
2026-02-04 23:22:02,348 [INFO] MainThread Output 4: name=rgb.width.function, type=tensor(int64), shape=[1]
2026-02-04 23:22:02,348 [INFO] MainThread Output 5: name=rgb.frame_id.function, type=tensor(int64), shape=[1]
2026-02-04 23:22:02,348 [INFO] MainThread Output 6: name=chroma.applier.function, type=tensor(float), shape=[1, 3, 'H', 'W']
2026-02-04 23:22:02,350 [INFO] MainThread [Main] starting thread Camera
2026-02-04 23:22:02,350 [INFO] Camera [Camera] start
2026-02-04 23:22:02,350 [INFO] MainThread [Main] starting thread Algos
2026-02-04 23:22:02,351 [DEBUG] Camera [gen] width=64 frame_id=0 wb=(1.001, 1.001, 1.001) offset=0.101
2026-02-04 23:22:02,352 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,352 [INFO] Algos [Algos] start
2026-02-04 23:22:02,352 [INFO] MainThread [Main] starting thread Coord
2026-02-04 23:22:02,352 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,353 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,353 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,356 [INFO] Coord [Coord] start
2026-02-04 23:22:02,356 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,357 [INFO] MainThread [Main] starting thread ISP
2026-02-04 23:22:02,357 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,358 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,358 [INFO] ISP [ISP] start
2026-02-04 23:22:02,359 [INFO] MainThread [Main] alive | counts: camera=0 algos=0 coord=0 isp=0
2026-02-04 23:22:02,364 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,367 [DEBUG] Camera [gen] width=64 frame_id=1 wb=(1.0019999999999998, 1.0019999999999998, 1.0019999999999998) offset=0.10200000000000001
2026-02-04 23:22:02,368 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,368 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,368 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,371 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,373 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,373 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,373 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,373 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,374 [DEBUG] Algos [algo→coord] put ok (size=2)
2026-02-04 23:22:02,379 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,379 [DEBUG] Camera [gen] width=64 frame_id=2 wb=(1.0029999999999997, 1.0029999999999997, 1.0029999999999997) offset=0.10300000000000001
2026-02-04 23:22:02,380 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,381 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,381 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,382 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,382 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,383 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,384 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,385 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,386 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,386 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,386 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,387 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,388 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,393 [DEBUG] Camera [gen] width=64 frame_id=3 wb=(1.0039999999999996, 1.0039999999999996, 1.0039999999999996) offset=0.10400000000000001
2026-02-04 23:22:02,393 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,394 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,394 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,395 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,395 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,396 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,396 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,397 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,397 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,400 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,400 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,400 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,401 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,401 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,401 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,402 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,407 [DEBUG] Camera [gen] width=64 frame_id=4 wb=(1.0049999999999994, 1.0049999999999994, 1.0049999999999994) offset=0.10500000000000001
2026-02-04 23:22:02,407 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,407 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,407 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,408 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,408 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,408 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,409 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,409 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,414 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,415 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,415 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,416 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,417 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,418 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,418 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,419 [DEBUG] Camera [gen] width=64 frame_id=5 wb=(1.0059999999999993, 1.0059999999999993, 1.0059999999999993) offset=0.10600000000000001
2026-02-04 23:22:02,420 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,420 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,421 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,421 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,422 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,423 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,423 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,424 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,429 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,433 [DEBUG] Camera [gen] width=64 frame_id=6 wb=(1.0069999999999992, 1.0069999999999992, 1.0069999999999992) offset=0.10700000000000001
2026-02-04 23:22:02,433 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,434 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,434 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,434 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,434 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,435 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,435 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,436 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,441 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,445 [DEBUG] Camera [gen] width=64 frame_id=7 wb=(1.0079999999999991, 1.0079999999999991, 1.0079999999999991) offset=0.10800000000000001
2026-02-04 23:22:02,446 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,446 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,447 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,447 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,447 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,448 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,449 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,450 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,453 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,454 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,454 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,455 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,455 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,456 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,457 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,459 [DEBUG] Camera [gen] width=64 frame_id=8 wb=(1.008999999999999, 1.008999999999999, 1.008999999999999) offset=0.10900000000000001
2026-02-04 23:22:02,459 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,460 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,460 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,460 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,460 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,461 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,461 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,462 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,462 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,462 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,463 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,463 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,463 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,464 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,469 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,471 [DEBUG] Camera [gen] width=64 frame_id=9 wb=(1.009999999999999, 1.009999999999999, 1.009999999999999) offset=0.11000000000000001
2026-02-04 23:22:02,471 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,471 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,472 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,472 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,472 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,472 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,473 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,475 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,475 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,475 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,475 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,476 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,476 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,477 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,482 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,483 [DEBUG] Camera [gen] width=64 frame_id=10 wb=(1.0109999999999988, 1.0109999999999988, 1.0109999999999988) offset=0.11100000000000002
2026-02-04 23:22:02,485 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,485 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,486 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,486 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,487 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,487 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,488 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,489 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,490 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,491 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,491 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,492 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,493 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,493 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,497 [DEBUG] Camera [gen] width=64 frame_id=11 wb=(1.0119999999999987, 1.0119999999999987, 1.0119999999999987) offset=0.11200000000000002
2026-02-04 23:22:02,497 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,497 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,498 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,498 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,498 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,498 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,499 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,500 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,505 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,509 [DEBUG] Camera [gen] width=64 frame_id=12 wb=(1.0129999999999986, 1.0129999999999986, 1.0129999999999986) offset=0.11300000000000002
2026-02-04 23:22:02,509 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,509 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,510 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,510 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,510 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,510 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,511 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,511 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,516 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,520 [DEBUG] Camera [gen] width=64 frame_id=13 wb=(1.0139999999999985, 1.0139999999999985, 1.0139999999999985) offset=0.11400000000000002
2026-02-04 23:22:02,520 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,521 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,521 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,521 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,521 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,522 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,522 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,523 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,528 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,532 [DEBUG] Camera [gen] width=64 frame_id=14 wb=(1.0149999999999983, 1.0149999999999983, 1.0149999999999983) offset=0.11500000000000002
2026-02-04 23:22:02,533 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,533 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,533 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,533 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,534 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,534 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,534 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,535 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,539 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,540 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,540 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,540 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,540 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,541 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,541 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,545 [DEBUG] Camera [gen] width=64 frame_id=15 wb=(1.0159999999999982, 1.0159999999999982, 1.0159999999999982) offset=0.11600000000000002
2026-02-04 23:22:02,545 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,546 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,546 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,546 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,546 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,547 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,547 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,548 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,553 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,557 [DEBUG] Camera [gen] width=64 frame_id=16 wb=(1.0169999999999981, 1.0169999999999981, 1.0169999999999981) offset=0.11700000000000002
2026-02-04 23:22:02,558 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,558 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,558 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,558 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,558 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,559 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,559 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,560 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,564 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,564 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,565 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,565 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,565 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,566 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,566 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,569 [DEBUG] Camera [gen] width=64 frame_id=17 wb=(1.017999999999998, 1.017999999999998, 1.017999999999998) offset=0.11800000000000002
2026-02-04 23:22:02,569 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,570 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,570 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,570 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,571 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,572 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,572 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,573 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,578 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,582 [DEBUG] Camera [gen] width=64 frame_id=18 wb=(1.018999999999998, 1.018999999999998, 1.018999999999998) offset=0.11900000000000002
2026-02-04 23:22:02,583 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,583 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,583 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,583 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,584 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,584 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,584 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,584 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,589 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,594 [DEBUG] Camera [gen] width=64 frame_id=19 wb=(1.0199999999999978, 1.0199999999999978, 1.0199999999999978) offset=0.12000000000000002
2026-02-04 23:22:02,595 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,595 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,596 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,596 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,596 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,596 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,596 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,597 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,602 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,602 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,602 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,602 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,603 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,603 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,604 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,606 [DEBUG] Camera [gen] width=64 frame_id=20 wb=(1.0209999999999977, 1.0209999999999977, 1.0209999999999977) offset=0.12100000000000002
2026-02-04 23:22:02,607 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,607 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,607 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,607 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,607 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,608 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,608 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,609 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,609 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,610 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,610 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,610 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,611 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,611 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,617 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,617 [DEBUG] Camera [gen] width=64 frame_id=21 wb=(1.0219999999999976, 1.0219999999999976, 1.0219999999999976) offset=0.12200000000000003
2026-02-04 23:22:02,618 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,618 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,618 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,618 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,619 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,619 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,619 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,623 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,623 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,623 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,623 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,624 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,624 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,624 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,629 [DEBUG] Camera [gen] width=64 frame_id=22 wb=(1.0229999999999975, 1.0229999999999975, 1.0229999999999975) offset=0.12300000000000003
2026-02-04 23:22:02,630 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,630 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,631 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,630 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,630 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,631 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,631 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,632 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,636 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,637 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,637 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,637 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,637 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,638 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,638 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,641 [DEBUG] Camera [gen] width=64 frame_id=23 wb=(1.0239999999999974, 1.0239999999999974, 1.0239999999999974) offset=0.12400000000000003
2026-02-04 23:22:02,642 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,642 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,642 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,643 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,643 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,643 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,643 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,644 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,644 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,644 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,644 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,645 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,645 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,646 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,651 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,653 [DEBUG] Camera [gen] width=64 frame_id=24 wb=(1.0249999999999972, 1.0249999999999972, 1.0249999999999972) offset=0.12500000000000003
2026-02-04 23:22:02,653 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,653 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,654 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,654 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,654 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,655 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,655 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,656 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,656 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,657 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,657 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,657 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,657 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,658 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,664 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,665 [DEBUG] Camera [gen] width=64 frame_id=25 wb=(1.0259999999999971, 1.0259999999999971, 1.0259999999999971) offset=0.12600000000000003
2026-02-04 23:22:02,665 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,665 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,665 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,666 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,666 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,666 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,667 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,669 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,670 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,670 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,670 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,671 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,671 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,672 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,677 [DEBUG] Camera [gen] width=64 frame_id=26 wb=(1.026999999999997, 1.026999999999997, 1.026999999999997) offset=0.12700000000000003
2026-02-04 23:22:02,677 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,677 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,677 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,678 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,678 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,678 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,679 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,679 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,683 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,684 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,684 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,684 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,685 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,685 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,686 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,689 [DEBUG] Camera [gen] width=64 frame_id=27 wb=(1.027999999999997, 1.027999999999997, 1.027999999999997) offset=0.12800000000000003
2026-02-04 23:22:02,689 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,689 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,689 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,689 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,690 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,690 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,691 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,691 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,691 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,691 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,691 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,692 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,692 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,693 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,698 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,700 [DEBUG] Camera [gen] width=64 frame_id=28 wb=(1.0289999999999968, 1.0289999999999968, 1.0289999999999968) offset=0.12900000000000003
2026-02-04 23:22:02,701 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,701 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,701 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,702 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,702 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,702 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,703 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,704 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,704 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,704 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,704 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,705 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,706 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,706 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,712 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,713 [DEBUG] Camera [gen] width=64 frame_id=29 wb=(1.0299999999999967, 1.0299999999999967, 1.0299999999999967) offset=0.13000000000000003
2026-02-04 23:22:02,713 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,713 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,713 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,713 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,714 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,714 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,715 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,718 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,718 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,718 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,719 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,719 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,719 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,720 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,724 [DEBUG] Camera [gen] width=64 frame_id=30 wb=(1.0309999999999966, 1.0309999999999966, 1.0309999999999966) offset=0.13100000000000003
2026-02-04 23:22:02,724 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,725 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,725 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,725 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,725 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,726 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,726 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,727 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,732 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,736 [DEBUG] Camera [gen] width=64 frame_id=31 wb=(1.0319999999999965, 1.0319999999999965, 1.0319999999999965) offset=0.13200000000000003
2026-02-04 23:22:02,736 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,737 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,737 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,737 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,737 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,738 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,738 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,739 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,743 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,743 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,744 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,744 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,744 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,745 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,745 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,748 [DEBUG] Camera [gen] width=64 frame_id=32 wb=(1.0329999999999964, 1.0329999999999964, 1.0329999999999964) offset=0.13300000000000003
2026-02-04 23:22:02,748 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,749 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,749 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,749 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,749 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,750 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,750 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,751 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,751 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,751 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,751 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,751 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,752 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,752 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,758 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,760 [DEBUG] Camera [gen] width=64 frame_id=33 wb=(1.0339999999999963, 1.0339999999999963, 1.0339999999999963) offset=0.13400000000000004
2026-02-04 23:22:02,761 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,761 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,761 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,761 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,762 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,762 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,763 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,763 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,763 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,763 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,764 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,764 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,765 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,765 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,770 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,772 [DEBUG] Camera [gen] width=64 frame_id=34 wb=(1.0349999999999961, 1.0349999999999961, 1.0349999999999961) offset=0.13500000000000004
2026-02-04 23:22:02,773 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,773 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,773 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,773 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,774 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,774 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,774 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,776 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,776 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,777 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,777 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,777 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,778 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,778 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,784 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,784 [DEBUG] Camera [gen] width=64 frame_id=35 wb=(1.035999999999996, 1.035999999999996, 1.035999999999996) offset=0.13600000000000004
2026-02-04 23:22:02,785 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,785 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,785 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,785 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,786 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,786 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,787 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,790 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,790 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,790 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,790 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,790 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,791 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,791 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,796 [DEBUG] Camera [gen] width=64 frame_id=36 wb=(1.036999999999996, 1.036999999999996, 1.036999999999996) offset=0.13700000000000004
2026-02-04 23:22:02,797 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,797 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,798 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,798 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,798 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,798 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,799 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,799 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,804 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,808 [DEBUG] Camera [gen] width=64 frame_id=37 wb=(1.0379999999999958, 1.0379999999999958, 1.0379999999999958) offset=0.13800000000000004
2026-02-04 23:22:02,808 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,809 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,809 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,809 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,809 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,810 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,810 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,810 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,815 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,820 [DEBUG] Camera [gen] width=64 frame_id=38 wb=(1.0389999999999957, 1.0389999999999957, 1.0389999999999957) offset=0.13900000000000004
2026-02-04 23:22:02,820 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,820 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,821 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,821 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,820 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,821 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,822 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,822 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,827 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,827 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,827 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,828 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,828 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,828 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,829 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,832 [DEBUG] Camera [gen] width=64 frame_id=39 wb=(1.0399999999999956, 1.0399999999999956, 1.0399999999999956) offset=0.14000000000000004
2026-02-04 23:22:02,832 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,832 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,833 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,833 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,833 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,833 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,833 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,834 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,834 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,834 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,835 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,835 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,836 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,836 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,842 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,843 [DEBUG] Camera [gen] width=64 frame_id=40 wb=(1.0409999999999955, 1.0409999999999955, 1.0409999999999955) offset=0.14100000000000004
2026-02-04 23:22:02,844 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,844 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,844 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,844 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,845 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,845 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,846 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,847 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,848 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,848 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,848 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,848 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,849 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,849 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,855 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,855 [DEBUG] Camera [gen] width=64 frame_id=41 wb=(1.0419999999999954, 1.0419999999999954, 1.0419999999999954) offset=0.14200000000000004
2026-02-04 23:22:02,856 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,856 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,856 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,856 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,857 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,857 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,857 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,861 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,861 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,861 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,861 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,862 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,862 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,863 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,867 [DEBUG] Camera [gen] width=64 frame_id=42 wb=(1.0429999999999953, 1.0429999999999953, 1.0429999999999953) offset=0.14300000000000004
2026-02-04 23:22:02,867 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,867 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,868 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,868 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,868 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,868 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,869 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,870 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,874 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,879 [DEBUG] Camera [gen] width=64 frame_id=43 wb=(1.0439999999999952, 1.0439999999999952, 1.0439999999999952) offset=0.14400000000000004
2026-02-04 23:22:02,880 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,880 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,881 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,880 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,880 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,881 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,881 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,882 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,886 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,887 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,887 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,887 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,887 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,888 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,888 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,894 [DEBUG] Camera [gen] width=64 frame_id=44 wb=(1.044999999999995, 1.044999999999995, 1.044999999999995) offset=0.14500000000000005
2026-02-04 23:22:02,894 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,894 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,895 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,895 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,895 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,896 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,896 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,896 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,901 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,901 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,901 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,902 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,902 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,902 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,903 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,906 [DEBUG] Camera [gen] width=64 frame_id=45 wb=(1.045999999999995, 1.045999999999995, 1.045999999999995) offset=0.14600000000000005
2026-02-04 23:22:02,906 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,907 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,907 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,907 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,907 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,908 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,908 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,908 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,914 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,918 [DEBUG] Camera [gen] width=64 frame_id=46 wb=(1.0469999999999948, 1.0469999999999948, 1.0469999999999948) offset=0.14700000000000005
2026-02-04 23:22:02,918 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,919 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,919 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,919 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,919 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,920 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,920 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,921 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,925 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,926 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,926 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,926 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,926 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,927 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,927 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,931 [DEBUG] Camera [gen] width=64 frame_id=47 wb=(1.0479999999999947, 1.0479999999999947, 1.0479999999999947) offset=0.14800000000000005
2026-02-04 23:22:02,931 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,931 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,932 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,932 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,932 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,932 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,933 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,933 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,939 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,943 [DEBUG] Camera [gen] width=64 frame_id=48 wb=(1.0489999999999946, 1.0489999999999946, 1.0489999999999946) offset=0.14900000000000005
2026-02-04 23:22:02,943 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,944 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,944 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,944 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,944 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,944 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,945 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,945 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,950 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,951 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,951 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,951 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,952 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,952 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,953 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,955 [DEBUG] Camera [gen] width=64 frame_id=49 wb=(1.0499999999999945, 1.0499999999999945, 1.0499999999999945) offset=0.15000000000000005
2026-02-04 23:22:02,956 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,956 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,956 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,956 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,957 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,957 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,958 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,958 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,958 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,958 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,958 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,959 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,959 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,960 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,965 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,967 [DEBUG] Camera [gen] width=64 frame_id=50 wb=(1.0509999999999944, 1.0509999999999944, 1.0509999999999944) offset=0.15100000000000005
2026-02-04 23:22:02,967 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,968 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,968 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,968 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,968 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,969 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,969 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,971 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,971 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,971 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,971 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,972 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,973 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,973 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,978 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,979 [DEBUG] Camera [gen] width=64 frame_id=51 wb=(1.0519999999999943, 1.0519999999999943, 1.0519999999999943) offset=0.15200000000000005
2026-02-04 23:22:02,979 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,979 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,980 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,980 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,980 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,980 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,981 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,984 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,984 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,985 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:02,985 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:02,985 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:02,986 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:02,986 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,991 [DEBUG] Camera [gen] width=64 frame_id=52 wb=(1.0529999999999942, 1.0529999999999942, 1.0529999999999942) offset=0.15300000000000005
2026-02-04 23:22:02,991 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,991 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:02,992 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:02,992 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:02,992 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:02,992 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,993 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:02,993 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:02,998 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,003 [DEBUG] Camera [gen] width=64 frame_id=53 wb=(1.053999999999994, 1.053999999999994, 1.053999999999994) offset=0.15400000000000005
2026-02-04 23:22:03,003 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,004 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,004 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,004 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,005 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,005 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,006 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,006 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,009 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,009 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,009 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,010 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,010 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,011 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,011 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,016 [DEBUG] Camera [gen] width=64 frame_id=54 wb=(1.054999999999994, 1.054999999999994, 1.054999999999994) offset=0.15500000000000005
2026-02-04 23:22:03,016 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,016 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,016 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,017 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,017 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,017 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,018 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,018 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,022 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,022 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,023 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,023 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,023 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,024 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,024 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,028 [DEBUG] Camera [gen] width=64 frame_id=55 wb=(1.0559999999999938, 1.0559999999999938, 1.0559999999999938) offset=0.15600000000000006
2026-02-04 23:22:03,028 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,028 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,028 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,029 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,029 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,029 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,030 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,030 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,030 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,030 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,031 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,031 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,032 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,032 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,038 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,039 [DEBUG] Camera [gen] width=64 frame_id=56 wb=(1.0569999999999937, 1.0569999999999937, 1.0569999999999937) offset=0.15700000000000006
2026-02-04 23:22:03,040 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,040 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,040 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,040 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,040 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,041 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,041 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,043 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,043 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,044 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,044 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,044 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,045 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,045 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,051 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,051 [DEBUG] Camera [gen] width=64 frame_id=57 wb=(1.0579999999999936, 1.0579999999999936, 1.0579999999999936) offset=0.15800000000000006
2026-02-04 23:22:03,051 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,052 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,052 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,052 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,052 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,052 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,053 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,056 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,057 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,057 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,057 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,058 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,058 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,059 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,062 [DEBUG] Camera [gen] width=64 frame_id=58 wb=(1.0589999999999935, 1.0589999999999935, 1.0589999999999935) offset=0.15900000000000006
2026-02-04 23:22:03,063 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,063 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,063 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,063 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,064 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,064 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,064 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,065 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,070 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,074 [DEBUG] Camera [gen] width=64 frame_id=59 wb=(1.0599999999999934, 1.0599999999999934, 1.0599999999999934) offset=0.16000000000000006
2026-02-04 23:22:03,075 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,075 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,075 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,075 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,075 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,076 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,076 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,077 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,081 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,081 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,082 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,082 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,082 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,083 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,083 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,086 [DEBUG] Camera [gen] width=64 frame_id=60 wb=(1.0609999999999933, 1.0609999999999933, 1.0609999999999933) offset=0.16100000000000006
2026-02-04 23:22:03,087 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,087 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,087 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,088 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,088 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,089 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,089 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,090 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,094 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,098 [DEBUG] Camera [gen] width=64 frame_id=61 wb=(1.0619999999999932, 1.0619999999999932, 1.0619999999999932) offset=0.16200000000000006
2026-02-04 23:22:03,099 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,099 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,099 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,099 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,100 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,100 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,100 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,101 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,106 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,110 [DEBUG] Camera [gen] width=64 frame_id=62 wb=(1.062999999999993, 1.062999999999993, 1.062999999999993) offset=0.16300000000000006
2026-02-04 23:22:03,111 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,111 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,111 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,111 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,111 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,112 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,112 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,113 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,117 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,117 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,118 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,118 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,118 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,119 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,119 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,123 [DEBUG] Camera [gen] width=64 frame_id=63 wb=(1.063999999999993, 1.063999999999993, 1.063999999999993) offset=0.16400000000000006
2026-02-04 23:22:03,123 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,123 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,123 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,123 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,124 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,124 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,125 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,125 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,130 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,134 [DEBUG] Camera [gen] width=64 frame_id=64 wb=(1.0649999999999928, 1.0649999999999928, 1.0649999999999928) offset=0.16500000000000006
2026-02-04 23:22:03,135 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,135 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,135 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,135 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,135 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,136 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,136 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,137 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,142 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,146 [DEBUG] Camera [gen] width=64 frame_id=65 wb=(1.0659999999999927, 1.0659999999999927, 1.0659999999999927) offset=0.16600000000000006
2026-02-04 23:22:03,147 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,147 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,147 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,147 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,147 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,148 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,148 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,149 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,153 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,154 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,154 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,155 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,155 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,156 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,157 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,158 [DEBUG] Camera [gen] width=64 frame_id=66 wb=(1.0669999999999926, 1.0669999999999926, 1.0669999999999926) offset=0.16700000000000007
2026-02-04 23:22:03,159 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,159 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,159 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,159 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,159 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,160 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,160 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,162 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,162 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,162 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,162 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,163 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,163 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,164 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,169 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,170 [DEBUG] Camera [gen] width=64 frame_id=67 wb=(1.0679999999999925, 1.0679999999999925, 1.0679999999999925) offset=0.16800000000000007
2026-02-04 23:22:03,170 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,171 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,171 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,171 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,172 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,172 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,173 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,175 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,175 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,175 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,175 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,176 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,176 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,177 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,182 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,182 [DEBUG] Camera [gen] width=64 frame_id=68 wb=(1.0689999999999924, 1.0689999999999924, 1.0689999999999924) offset=0.16900000000000007
2026-02-04 23:22:03,183 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,183 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,183 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,183 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,183 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,184 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,184 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,188 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,188 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,188 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,189 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,189 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,190 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,190 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,194 [DEBUG] Camera [gen] width=64 frame_id=69 wb=(1.0699999999999923, 1.0699999999999923, 1.0699999999999923) offset=0.17000000000000007
2026-02-04 23:22:03,194 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,194 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,195 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,195 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,195 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,195 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,196 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,197 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,201 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,206 [DEBUG] Camera [gen] width=64 frame_id=70 wb=(1.0709999999999922, 1.0709999999999922, 1.0709999999999922) offset=0.17100000000000007
2026-02-04 23:22:03,206 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,206 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,206 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,207 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,207 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,207 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,208 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,208 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,213 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,213 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,213 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,213 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,214 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,214 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,215 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,218 [DEBUG] Camera [gen] width=64 frame_id=71 wb=(1.071999999999992, 1.071999999999992, 1.071999999999992) offset=0.17200000000000007
2026-02-04 23:22:03,218 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,218 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,219 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,219 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,219 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,219 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,220 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,220 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,221 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,221 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,221 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,222 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,222 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,223 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,228 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,229 [DEBUG] Camera [gen] width=64 frame_id=72 wb=(1.072999999999992, 1.072999999999992, 1.072999999999992) offset=0.17300000000000007
2026-02-04 23:22:03,230 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,230 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,230 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,230 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,231 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,231 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,231 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,234 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,234 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,234 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,234 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,235 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,235 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,236 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,241 [DEBUG] Camera [gen] width=64 frame_id=73 wb=(1.0739999999999919, 1.0739999999999919, 1.0739999999999919) offset=0.17400000000000007
2026-02-04 23:22:03,241 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,242 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,242 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,242 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,242 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,243 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,243 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,243 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,247 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,247 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,247 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,248 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,248 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,249 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,249 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,253 [DEBUG] Camera [gen] width=64 frame_id=74 wb=(1.0749999999999917, 1.0749999999999917, 1.0749999999999917) offset=0.17500000000000007
2026-02-04 23:22:03,253 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,254 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,254 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,254 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,255 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,255 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,255 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,256 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,260 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,265 [DEBUG] Camera [gen] width=64 frame_id=75 wb=(1.0759999999999916, 1.0759999999999916, 1.0759999999999916) offset=0.17600000000000007
2026-02-04 23:22:03,266 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,266 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,266 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,266 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,266 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,267 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,267 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,268 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,272 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,272 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,272 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,273 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,273 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,274 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,274 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,277 [DEBUG] Camera [gen] width=64 frame_id=76 wb=(1.0769999999999915, 1.0769999999999915, 1.0769999999999915) offset=0.17700000000000007
2026-02-04 23:22:03,278 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,278 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,278 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,278 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,279 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,279 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,279 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,280 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,285 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,289 [DEBUG] Camera [gen] width=64 frame_id=77 wb=(1.0779999999999914, 1.0779999999999914, 1.0779999999999914) offset=0.17800000000000007
2026-02-04 23:22:03,289 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,290 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,290 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,290 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,290 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,290 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,291 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,291 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,296 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,301 [DEBUG] Camera [gen] width=64 frame_id=78 wb=(1.0789999999999913, 1.0789999999999913, 1.0789999999999913) offset=0.17900000000000008
2026-02-04 23:22:03,301 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,301 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,301 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,302 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,302 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,302 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,303 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,303 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,308 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,313 [DEBUG] Camera [gen] width=64 frame_id=79 wb=(1.0799999999999912, 1.0799999999999912, 1.0799999999999912) offset=0.18000000000000008
2026-02-04 23:22:03,313 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,313 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,314 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,314 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,314 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,314 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,315 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,315 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,319 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,319 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,319 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,319 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,320 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,320 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,321 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,325 [DEBUG] Camera [gen] width=64 frame_id=80 wb=(1.080999999999991, 1.080999999999991, 1.080999999999991) offset=0.18100000000000008
2026-02-04 23:22:03,325 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,325 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,325 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,326 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,326 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,326 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,327 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,327 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,332 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,337 [DEBUG] Camera [gen] width=64 frame_id=81 wb=(1.081999999999991, 1.081999999999991, 1.081999999999991) offset=0.18200000000000008
2026-02-04 23:22:03,337 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,337 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,338 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,338 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,339 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,339 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,339 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,340 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,344 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,344 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,344 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,344 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,345 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,345 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,346 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,349 [DEBUG] Camera [gen] width=64 frame_id=82 wb=(1.0829999999999909, 1.0829999999999909, 1.0829999999999909) offset=0.18300000000000008
2026-02-04 23:22:03,350 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,350 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,350 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,350 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,351 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,351 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,352 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,352 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,357 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,361 [DEBUG] Camera [gen] width=64 frame_id=83 wb=(1.0839999999999907, 1.0839999999999907, 1.0839999999999907) offset=0.18400000000000008
2026-02-04 23:22:03,362 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,362 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,362 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,362 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,363 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,363 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,363 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,364 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,368 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,373 [DEBUG] Camera [gen] width=64 frame_id=84 wb=(1.0849999999999906, 1.0849999999999906, 1.0849999999999906) offset=0.18500000000000008
2026-02-04 23:22:03,373 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,373 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,373 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,374 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,374 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,374 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,375 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,375 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,380 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,385 [DEBUG] Camera [gen] width=64 frame_id=85 wb=(1.0859999999999905, 1.0859999999999905, 1.0859999999999905) offset=0.18600000000000008
2026-02-04 23:22:03,385 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,385 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,385 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,385 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,386 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,386 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,387 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,387 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,391 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,392 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,392 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,392 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,392 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,393 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,393 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,397 [DEBUG] Camera [gen] width=64 frame_id=86 wb=(1.0869999999999904, 1.0869999999999904, 1.0869999999999904) offset=0.18700000000000008
2026-02-04 23:22:03,397 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,397 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,397 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,398 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,398 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,398 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,399 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,399 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,404 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,408 [DEBUG] Camera [gen] width=64 frame_id=87 wb=(1.0879999999999903, 1.0879999999999903, 1.0879999999999903) offset=0.18800000000000008
2026-02-04 23:22:03,409 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,409 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,409 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,409 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,409 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,410 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,410 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,411 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,416 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,420 [DEBUG] Camera [gen] width=64 frame_id=88 wb=(1.0889999999999902, 1.0889999999999902, 1.0889999999999902) offset=0.18900000000000008
2026-02-04 23:22:03,420 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,421 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,421 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,421 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,421 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,422 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,422 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,423 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,427 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,427 [DEBUG] Coord [Coord] merged keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,427 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,427 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,428 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,428 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,429 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,432 [DEBUG] Camera [gen] width=64 frame_id=89 wb=(1.08999999999999, 1.08999999999999, 1.08999999999999) offset=0.19000000000000009
2026-02-04 23:22:03,432 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,433 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,434 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,435 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,435 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,436 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,437 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,439 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,443 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,447 [DEBUG] Camera [gen] width=64 frame_id=90 wb=(1.09099999999999, 1.09099999999999, 1.09099999999999) offset=0.1910000000000001
2026-02-04 23:22:03,447 [DEBUG] Camera [gen] out keys=['image_desc.input.image', 'image_desc.input.width', 'image_desc.input.frame_id', 'blacklevel.offset', 'image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,448 [DEBUG] Camera [cam→algo] put ok (size=1)
2026-02-04 23:22:03,449 [DEBUG] Algos [Algos] got msg type=dict
2026-02-04 23:22:03,449 [DEBUG] Coord [Coord] waiting for both camera+algo
2026-02-04 23:22:03,450 [DEBUG] Camera [cam→coord] put ok (size=1)
2026-02-04 23:22:03,451 [DEBUG] Algos [filter_feed] in=8 kept=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,452 [DEBUG] Algos [Algos] feed keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function']
2026-02-04 23:22:03,454 [DEBUG] Algos [algo→coord] put ok (size=1)
2026-02-04 23:22:03,457 [DEBUG] Coord [map_outs] keys=['image_desc.width.function', 'bayer2cfa.cfa_onehot.function', 'awb.wb_gains.function', 'ccm.ccm.normalized.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'rgb.rgb_out.function', 'rgb.height.function', 'rgb.width.function', 'rgb.frame_id.function', 'yuv.rgb2yuv_matrix.normalized.function', 'yuv.rgb2yuv_matrix.function', 'chroma.applier.function', 'chroma.subsample_scale.function']
^C2026-02-04 23:22:03,458 [DEBUG] Coord [coord→isp] put ok (size=1)
2026-02-04 23:22:03,459 [INFO] MainThread [Main] Ctrl+C received, stopping...
2026-02-04 23:22:03,459 [DEBUG] ISP [ISP] got feed type=dict
2026-02-04 23:22:03,460 [INFO] MainThread [Main] joining Camera
2026-02-04 23:22:03,460 [DEBUG] ISP [ISP] filtered keys=['image_desc.input.image.function', 'image_desc.input.width.function', 'image_desc.input.frame_id.function', 'blacklevel.offset.function', 'awb.wb_gains.function', 'ccm.ccm.function', 'tonemap.tonemap_curve.function', 'gamma.gamma_value.function', 'yuv.rgb2yuv_matrix.function', 'chroma.subsample_scale.function']
2026-02-04 23:22:03,461 [INFO] Coord [Coord] stop
2026-02-04 23:22:03,462 [DEBUG] ISP [ISP] outputs count=7
2026-02-04 23:22:03,462 [INFO] ISP [ISP] stop
2026-02-04 23:22:03,462 [INFO] Camera [Camera] stop
2026-02-04 23:22:03,463 [INFO] MainThread [Main] joining Algos
2026-02-04 23:22:03,555 [INFO] Algos [Algos] stop
2026-02-04 23:22:03,556 [INFO] MainThread [Main] joining Coord
2026-02-04 23:22:03,557 [INFO] MainThread [Main] joining ISP

=== Timing summary ===
camera: min=0.010986s max=0.016560s mean=0.012207s count=91
algos: min=0.001187s max=0.020297s mean=0.012099s count=91
coord: min=0.001589s max=0.008166s mean=0.004853s count=227
isp: min=0.006456s max=0.047528s mean=0.019342s count=57
