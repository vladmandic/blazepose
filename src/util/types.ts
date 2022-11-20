import type * as tf from '@tensorflow/tfjs';

export type Box = [number, number, number, number];
export type Point = [number, number, number];
export type Keypoint = { score: number, presence: number, part: string, position: Point };
export type GraphModel = tf.GraphModel;
export type Tensor = tf.Tensor;
export type Tensor1D = tf.Tensor1D;
export type Tensor2D = tf.Tensor2D;
export type Config = any; // eslint-disable-line @typescript-eslint/no-explicit-any
export type BodyResult = any; // eslint-disable-line @typescript-eslint/no-explicit-any
