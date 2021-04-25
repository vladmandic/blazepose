# BlazePose: Body Segmentation for TFJS and NodeJS

Models included in `/model/*` were converted to TFJS Graph model format from the original repository  
Models descriptors and signature have been additionally parsed for readability

Actual model parsing implementation in `/src` does not follow original  
and is implemented using native TFJS ops and optimized for JavaScript execution

<br><hr><br>

## Work in Progress

<br><hr><br>

## Models

### Detector

```shell
# node signature.js model/blazepose-detect.json
```

```js
DATA:  inputs: [ { name: 'input:0', dtype: 'DT_FLOAT', shape: [ 1, 128, 128, 3 ] } ]
DATA:  outputs: [
  { id: 0, name: 'classificators:0', dytpe: 'DT_FLOAT', shape: [ 1, 896, 1 ] },
  { id: 1, name: 'regressors:0', dytpe: 'DT_FLOAT', shape: [ 1, 896, 12 ] },
]
```

### Full Body Keypoints

```shell
# node signature.js model/blazepose-full.json
```

```js
DATA:  inputs: [ { name: 'input_1:0', dtype: 'DT_FLOAT', shape: [ 1, 256, 256, 3 ] } ]
DATA:  outputs: [
  { id: 0, name: 'ld_3d:0', dytpe: 'DT_FLOAT', shape: [ 1, 195 ] },
  { id: 1, name: 'output_segmentation:0', dytpe: 'DT_FLOAT', shape: [ 1, 128, 128, 1 ] },
  { id: 2, name: 'output_poseflag:0', dytpe: 'DT_FLOAT', shape: [ 1, 1 ] },
]
```

### Upper Body Keypoints

```shell
# node signature.js model/blazepose-upper.json
```

```js
DATA:  inputs: [ { name: 'input_1:0', dtype: 'DT_FLOAT', shape: [ 1, 256, 256, 3 ] } ]
DATA:  outputs: [
  { id: 0, name: 'output_poseflag:0', dytpe: 'DT_FLOAT', shape: [ 1, 1 ] },
  { id: 1, name: 'output_segmentation:0', dytpe: 'DT_FLOAT', shape: [ 1, 128, 128, 1 ] },
  { id: 2, name: 'ld_3d:0', dytpe: 'DT_FLOAT', shape: [ 1, 155 ] },
]
```

<br><hr><br>

## Credits & Links

- Blog: <https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html>
- Docs: <https://google.github.io/mediapipe/solutions/pose>
- Paper: <https://arxiv.org/abs/2006.10204>
- Card: <https://drive.google.com/file/d/1UqfMdZ4sqioHULu6E-W0gp6JCcXAZPGK/view>
