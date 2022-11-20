import * as tf from '@tensorflow/tfjs';
import { log, join } from './util/util';
import type { Box, GraphModel, Config, Tensor } from './util/types';

let model: GraphModel | null = null;
let inputSize = 0;
let padding: [number, number][] = [[0, 0], [0, 0], [0, 0], [0, 0]];

export async function loadDetect(config: Config): Promise<GraphModel> {
  if (!model && config.body.detector?.modelPath || '') {
    model = await tf.loadGraphModel(join(config.modelBasePath, config.body.detector?.modelPath || '')) as unknown as GraphModel;
    const inputs = Object.values(model.modelSignature['inputs']);
    inputSize = Array.isArray(inputs) ? parseInt(inputs[0].tensorShape.dim[1].size) : 0;
    if (!model || !model['modelUrl']) log('load model failed:', config.object.modelPath);
    else if (config.debug) log('load model:', model['modelUrl']);
  } else if (config.debug && model) log('cached model:', model['modelUrl']);
  return model as GraphModel;
}

function prepareImage(input: Tensor): Tensor {
  if (!input.shape || !input.shape[1] || !input.shape[2]) return input;
  padding = [
    [0, 0], // dont touch batch
    [input.shape[2] > input.shape[1] ? Math.trunc((input.shape[2] - input.shape[1]) / 2) : 0, input.shape[2] > input.shape[1] ? Math.trunc((input.shape[2] - input.shape[1]) / 2) : 0], // height before&after
    [input.shape[1] > input.shape[2] ? Math.trunc((input.shape[1] - input.shape[2]) / 2) : 0, input.shape[1] > input.shape[2] ? Math.trunc((input.shape[1] - input.shape[2]) / 2) : 0], // width before&after
    [0, 0], // dont touch rbg
  ];
  const padT = tf.pad(input as tf.Tensor4D, padding);
  const resizeT = tf.image.resizeBilinear(padT as tf.Tensor4D, [inputSize, inputSize]);
  tf.dispose(padT);
  return resizeT;
}

async function decodeResults(res: Tensor, config: Config): Promise<Box[]> {
  log('decodeResults', { res, config });
  const box: Box = [0, 0, 1, 1]; // [x1,y1,x2,y2]
  const boxes: Box[] = [];
  boxes.push(box);
  return boxes;
}

export async function runDetect(input: Tensor, config: Config, outputSize: [number, number]): Promise<Box[]> {
  const squareT = prepareImage(input);
  const resultT = await model?.execute(squareT) as Tensor; // 1, 2254, 13
  const boxes = await decodeResults(resultT, config);
  const rescaled: Box[] = boxes.map((box) => [box[0] * outputSize[0], box[1] * outputSize[1], box[2] * outputSize[0], box[3] * outputSize[1]]);
  tf.dispose([squareT, resultT]);
  return rescaled;
}
