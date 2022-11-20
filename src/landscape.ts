/**
 * BlazePose model implementation
 */

import * as tf from '@tensorflow/tfjs';
import { log, join } from './util/util';
import * as coords from './blazeposecoords';
import type { Point, Box, Tensor, GraphModel, Config, BodyResult, Keypoint } from './util/types';

let model: GraphModel | null = null;
let inputSize = 0;
const outputNodes = ['ld_3d', 'activation_segmentation', 'activation_heatmap', 'world_3d', 'output_poseflag'];
const depth = 5; // each points has x,y,z,visibility,presence

export async function loadLandscape(config: Config): Promise<GraphModel> {
  if (!model) {
    model = await tf.loadGraphModel(join(config.modelBasePath, config.body.modelPath || '')) as unknown as GraphModel;
    const inputs = Object.values(model.modelSignature['inputs']);
    inputSize = Array.isArray(inputs) ? parseInt(inputs[0].tensorShape.dim[1].size) : 0;
    if (!model || !model['modelUrl']) log('load model failed:', config.object.modelPath);
    else if (config.debug) log('load model:', model['modelUrl']);
  } else if (config.debug) log('cached model:', model['modelUrl']);
  return model;
}

function calculateBoxes(_config: Config, keypoints: Array<Keypoint>): Box {
  const x = keypoints.map((a) => a.position[0]);
  const y = keypoints.map((a) => a.position[1]);
  const box: Box = [Math.min(...x), Math.min(...y), Math.max(...x) - Math.min(...x), Math.max(...y) - Math.min(...y)];
  return box;
}

function rescaleKeypoint(point: Point, box: Box, outputSize: [number, number]): Point {
  const rescale: Point = [
    outputSize[0] * (point[0] / inputSize * (box[2] - box[0]) + box[0]),
    // outputSize[1] * (point[1] / inputSize / (box[3] - box[1]) + box[1]) - padding[3][0],
    outputSize[1] * (point[1] / inputSize * (box[3] - box[1]) + box[1]),
    point[2] / inputSize,
  ];
  return rescale;
}

export async function runLandscape(input: Tensor, box: Box, config: Config, outputSize: [number, number]): Promise<BodyResult> {
  const t: Record<string, Tensor> = {};
  const cropBox = [box[1], box[0], box[3], box[2]]; // invert x/y
  t.input = tf.image.cropAndResize(input as tf.Tensor4D, [cropBox], [0], [inputSize, inputSize]); // crop to detected box
  [t.ld/* 1,195=39*5 */, t.segmentation/* 1,256,256,1 */, t.heatmap/* 1,64,64,39 */, t.world/* 1,117 */, t.poseflag/* 1,1 */] = await model?.execute(t.input, outputNodes) as Tensor[]; // run model
  const points = await t.ld.data();
  const keypoints: Keypoint[] = [];
  for (let i = 0; i < points.length / depth; i++) {
    const part = coords.kpt[i];
    const score = Math.round(100 / (1 + Math.exp(-points[depth * i + 3]))) / 100; // normally this is from tf.sigmoid but no point of running sigmoid on full array which has coords as well
    const presence = Math.round(points[depth * i + 4] / 0.2) / 100;
    const point: Point = [points[depth * i + 0], points[depth * i + 1], points[depth * i + 2]];
    const position = rescaleKeypoint(point, box, outputSize);
    // if (score >= (config?.body?.minConfidence || 0) && presence >= (config?.body?.minConfidence || 0))
    keypoints.push({ score, presence, part, position });
  }
  const keypointBox = calculateBoxes(config, keypoints); // now find boxes based on rescaled keypoints
  Object.keys(t).forEach((tensor) => tf.dispose(t[tensor]));
  const annotations: Record<string, Point[][]> = {};
  for (const [name, indexes] of Object.entries(coords.connected)) { // lookup annoted body parts and fill them
    const pt: Array<Point[]> = [];
    for (let i = 0; i < indexes.length - 1; i++) {
      const pt0 = keypoints.find((kpt) => kpt.part === indexes[i]);
      const pt1 = keypoints.find((kpt) => kpt.part === indexes[i + 1]);
      if (!pt0 || !pt1) continue;
      pt.push([pt0.position, pt1.position]);
    }
    annotations[name] = pt;
  }
  const result = { id: 0, score: 0, keypoints, annotations, box: keypointBox };
  return result;
}
