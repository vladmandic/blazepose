const tf = require('@tensorflow/tfjs-node');
const annotations = require('./annotations');

const models = [];

const modelOptions = {
  detectorPath: 'file://model/blazepose-detect.json',
  modelFullPath: 'file://model/blazepose-full.json',
  modelUpperPath: 'file://model/blazepose-upper.json',
  minScore: 0.1,
};
const depth = 5; // each points has x,y,z,visibility,presence

async function load() {
  if (models.length === 0) {
    models.push(await tf.loadGraphModel(modelOptions.detectorPath));
    models.push(await tf.loadGraphModel(modelOptions.modelFullPath));
    models.push(await tf.loadGraphModel(modelOptions.modelUpperPath));
  }
  return models;
}

async function runDetect(input) {
  /* keypoints are *maybe*
  - face box
  - middle point between hips
  - size of the circle circumscribing the whole person,
  - angle between the lines connecting the two mid-shoulder and mid-hip points
  */
  const normalize = input.div(127.5).sub(1);
  const resize = tf.image.resizeBilinear(normalize, [128, 128]);
  normalize.dispose();
  const detectT = await models[0].predict(resize);
  resize.dispose();

  const argMax = detectT.find((t) => t.size === 896).argMax(1); // location of best score
  const max = argMax.dataSync()[0];
  const points = detectT.find((t) => t.size === 10752).arraySync()[0]; // array of 896 possible x 12 values per entry which are likely 6 points
  argMax.dispose();
  detectT.forEach((t) => t.dispose());

  const keypoints = [];
  for (let i = 0; i < points[max].length; i++) {
    keypoints.push({
      id: i / 2,
      position: {
        x: (Math.trunc(input.shape[2] * points[max][i] / 128)),
        y: (Math.trunc(input.shape[1] * points[max][++i] / 128)),
      },
    });
  }
  return { name: 'detect', keypoints };
}

async function runFull(input) {
  const normalize = input.div(127.5).sub(1);
  const resize = tf.image.resizeBilinear(normalize, [256, 256]);
  normalize.dispose();
  const resT = await models[1].predict(resize); // blazepose-full
  resize.dispose();

  const points = resT.find((t) => t.size === 195).dataSync(); // order of output tensors may change between models, full has 195 and upper has 155 items
  resT.forEach((t) => t.dispose());
  let totalScore = 0;

  const allKeypoints = [];
  for (let i = 0; i < points.length / depth; i++) {
    const visibility = (100 - Math.trunc(100 / (1 + Math.exp(points[depth * i + 3])))) / 100; // reverse sigmoid value
    const presence = (100 - Math.trunc(100 / (1 + Math.exp(points[depth * i + 4])))) / 100; // reverse sigmoid value
    totalScore += Math.min(visibility, presence);
    allKeypoints.push({
      id: i,
      part: annotations.full[i],
      position: {
        x: Math.trunc(input.shape[2] * points[depth * i + 0] / 256), // return normalized x value istead of 0..255
        y: Math.trunc(input.shape[1] * points[depth * i + 1] / 256), // return normalized y value istead of 0..255
        z: Math.trunc(points[depth * i + 2]) + 0, // fix negative zero
      },
      score: Math.min(visibility, presence),
    });
  }

  const avgScore = totalScore / allKeypoints.length;
  const keypoints = allKeypoints.filter((a) => a.score > modelOptions.minScore);
  const visibileScore = totalScore / keypoints.length;
  return { name: 'full', keypoints, visibleParts: keypoints.length, visibileScore, missingParts: allKeypoints.length - keypoints.length, avgScore };
}

async function runUpper(input) {
  const normalize = input.div(127.5).sub(1);
  const resize = tf.image.resizeBilinear(normalize, [256, 256]);
  normalize.dispose();
  const resT = await models[2].predict(resize); // blazepose-full
  resize.dispose();

  const points = resT.find((t) => t.size === 155).dataSync(); // order of output tensors may change between models, full has 195 and upper has 155 items
  resT.forEach((t) => t.dispose());
  let totalScore = 0;

  const allKeypoints = [];
  for (let i = 0; i < points.length / depth; i++) {
    const visibility = (100 - Math.trunc(100 / (1 + Math.exp(points[depth * i + 3])))) / 100; // reverse sigmoid value
    const presence = (100 - Math.trunc(100 / (1 + Math.exp(points[depth * i + 4])))) / 100; // reverse sigmoid value
    totalScore += Math.min(visibility, presence);
    allKeypoints.push({
      id: i,
      part: annotations.upper[i],
      position: {
        x: Math.trunc(input.shape[2] * points[depth * i + 0] / 256), // return normalized x value istead of 0..255
        y: Math.trunc(input.shape[1] * points[depth * i + 1] / 256), // return normalized y value istead of 0..255
        z: Math.trunc(points[depth * i + 2]) + 0, // fix negative zero
      },
      score: Math.min(visibility, presence),
    });
  }

  const avgScore = totalScore / allKeypoints.length;
  const keypoints = allKeypoints.filter((a) => a.score > modelOptions.minScore);
  const visibileScore = totalScore / keypoints.length;
  return { name: 'upper', keypoints, visibleParts: keypoints.length, visibileScore, missingParts: allKeypoints.length - keypoints.length, avgScore };
}

async function predict(input, options) {
  const results = [];
  if (options?.detect) results.push(await runDetect(input));
  if (options?.full) results.push(await runFull(input));
  if (options?.upper) results.push(await runUpper(input));
  return results;
}

exports.predict = predict;
exports.load = load;
