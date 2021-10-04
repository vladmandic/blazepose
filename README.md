# BlazePose: Body Segmentation for TFJS

*Updated for BlazePose v2 variations*

Included models:

- BlazePose Detector
- BlazePose Lite, Full & Heavy variations

Models included in `/model/*` were manually converted and quantized to TFJS Graph model format  
from the original repository and are not identical to TFJS Graph models published on [TF Hub](https://tfhub.dev/s?q=blazepose)  

Models descriptors and signature have been additionally parsed for readability

<br>

## Implementation

Actual model parsing implementation in `/src/blazepose.ts` does not follow [reference implementation](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection)

BlazePose is a two-phase model:

- Detector
- Pose analysis

Pose analysis is a fast, clean and accurate model  
However, detector is much slower and requires a lot of post-processing - thus its usage has been eliminated from this implementation  

Instead, this implementation prepares *virtual* detected tensor padded and resized to fit pose analysis model

Implementation is in `src/blazepose.ts` with keypoint definitions and annotations in `src/blazeposecoords.ts`  
The rest of sources in `src/util` is for simplicity of usage and testing only  

Ideal implementation includes additional results caching and temporal interpolation with smoothing functionality  
See <https://github.com/vladmandic/human> for details  

## Result

Implementation defines results a:

- `keypoints`: array of 39 detected keypoints
  - `part`: body part label
  - `position`: [x, y, z] normalized to input size
  - `postitionRaw`: [x, y, z] normalized to 0..1
  - `score`: detection score for the body part
- `box`: [x, y, width, height] box around detected body normalized to input size
- `boxRaw`: [x, y, width, height] box around detected body normalized to 0..1
- `annotations`: annotated arrays of points to help define higher level entities such as `arm` or `leg`
- `score`: average score for body

## Demo

Demo app that initializes webcam and uses `BlazePose` is in `demo/index.html`  
To run image analysis instead of default analysis on webcam input, edit `demo/index.js`
Example:

```js
await detectImage('daz3d-ella.jpg');
```

## Credits & Links

- Blog: <https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html>
- Docs: <https://google.github.io/mediapipe/solutions/pose>
- Paper: <https://arxiv.org/abs/2006.10204>
- Card: <https://drive.google.com/file/d/1UqfMdZ4sqioHULu6E-W0gp6JCcXAZPGK/view>

## Todo

- Enhance virtual tensor with virtual box around previous body center  
  For details, see disabled code in `calculateBoxes` method
- HeatMap processing
