# Auto Exposure Microblock

This microblock provides auto exposure functionality for the ISP pipeline with two different technical approaches that fit different scenarios.

## Overview

The auto exposure microblock follows the standard ISP pattern:
- **Algorithm (algo)**: Calculates exposure statistics and coefficients
- **Applier**: Uses the calculated coefficients to apply exposure compensation

## Two Technical Approaches

### 1. Statistics-Based Auto Exposure (`autoexposure_stats`)

**Use Case**: Direct RGB image analysis
**Input**: RGB image from demosaic stage
**Position**: After demosaic, before AWB

**How it works**:
- Calculates mean brightness directly from RGB image
- Computes exposure compensation based on RGB statistics
- Suitable for scenarios where you want to analyze the full color image

**Pipeline Integration**:
```json
"demosaic": {"class":"demosaic_mhc","version":"v1","inputs":["blacklevel", "bayer2cfa"]},
"autoexposure_stats": {"class":"autoexposure_stats","version":"v1","inputs":["demosaic"]},
"awb": {"class":"wb_avg_v1","version":"v1","inputs":["autoexposure_stats"]},
```

### 2. YUV-Based Auto Exposure (`autoexposure_yuv`)

**Use Case**: Luminance-based analysis
**Input**: YUV image from YUV conversion stage
**Position**: After YUV conversion

**How it works**:
- Extracts Y (luminance) channel from YUV image
- Calculates exposure based on luminance statistics
- Suitable for scenarios where you want to analyze brightness independently of color

**Pipeline Integration**:
```json
"yuv": {"class":"yuvconvert_base","version":"v0","inputs":["gamma"]},
"autoexposure_yuv": {"class":"autoexposure_yuv","version":"v1","inputs":["yuv"]},
```

## Implementation Details

### Algorithm Stage (`build_algo`)

The algorithm stage calculates exposure statistics and coefficients:

**Outputs**:
- `stats`: Calculated brightness statistics
- `exposure_value`: Exposure compensation value
- `gain`: Gain adjustment value

**Processing**:
1. Analyzes input image (RGB or YUV depending on approach)
2. Calculates mean brightness
3. Computes optimal exposure compensation
4. Outputs coefficients for applier stage

### Applier Stage (`build_applier`)

The applier stage uses the calculated coefficients:

**Inputs**:
- Input image from previous stage
- Exposure coefficients from algorithm stage

**Outputs**:
- Exposure-compensated image

**Processing**:
1. Receives coefficients from algorithm stage
2. Applies exposure compensation to input image
3. Outputs adjusted image

## Usage

### Build with Statistics-Based Auto Exposure

```bash
python build_all.py pipeline_stats_ae.json --mode algo
python build_all.py pipeline_stats_ae.json --mode applier
```

### Build with YUV-Based Auto Exposure

```bash
python build_all.py pipeline_yuv_ae.json --mode algo
python build_all.py pipeline_yuv_ae.json --mode applier
```

## Architecture Benefits

1. **Separation of Concerns**: Algorithm and applier are separate, allowing different optimization strategies
2. **Flexibility**: Choose the approach that fits your scenario
3. **Reusability**: Same applier logic works for both approaches
4. **Testability**: Each stage can be tested independently

## Technical Differences

| Aspect | Stats-Based | YUV-Based |
|--------|-------------|------------|
| Input Format | RGB | YUV |
| Analysis Method | Full color image | Luminance channel only |
| Position in Pipeline | After demosaic | After YUV conversion |
| Use Case | Color-aware exposure | Brightness-only exposure |
| Computational Cost | Higher (3 channels) | Lower (1 channel) |

## Future Enhancements

- Add histogram-based exposure calculation
- Implement adaptive exposure algorithms
- Add support for different exposure targets
- Implement exposure smoothing and temporal filtering