/**
 * BlazePose model implementation
 */

import type { GraphModel, Tensor, Config, Box, BodyResult } from './util/types';
import { loadDetect, runDetect } from './detect';
import { loadLandscape, runLandscape } from './landscape';

export async function load(config: Config): Promise<[GraphModel | null, GraphModel | null]> {
  const modelDetect = config.body.detector.enabled ? await loadDetect(config) : null;
  const modelLandscape = config.body.enabled ? await loadLandscape(config) : null;
  return [modelDetect, modelLandscape];
}

export async function predict(input: Tensor, config: Config) {
  const outputSize: [number, number] = [input.shape[2] || 0, input.shape[1] || 0];
  const boxes: Box[] = config.body.detector.enabled
    ? await runDetect(input, config, outputSize) // get actual detection boxes
    : [[0, 0, 1, 1]]; // use full input
  const results: BodyResult[] = [];
  for (const box of boxes) {
    const result = await runLandscape(input, box, config, outputSize);
    if (result) results.push(result);
  }
  return results;
}
