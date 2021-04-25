const fs = require('fs');
const path = require('path');
const process = require('process');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');
const canvas = require('canvas');

const modelOptions = {
  detectorPath: 'file://models/blazepose-detector.json',
  modelPath: 'file://models/blazepose-full.json',
  minScore: 0.2,
};

const bodyParts = ['head', 'neck', 'rightShoulder', 'rightElbow', 'rightWrist', 'chest', 'leftShoulder', 'leftElbow', 'leftWrist', 'pelvis', 'rightHip', 'rightKnee', 'rightAnkle', 'leftHip', 'leftKnee', 'leftAnkle'];

// save image with processed results
async function saveImage(res, img) {
  // create canvas
  const c = new canvas.Canvas(img.inputShape[1], img.inputShape[0]);
  const ctx = c.getContext('2d');

  // load and draw original image
  const original = await canvas.loadImage(img.fileName);
  ctx.drawImage(original, 0, 0, c.width, c.height);
  // const fontSize = Math.trunc(c.width / 50);
  const fontSize = Math.round((c.width * c.height) ** (1 / 2) / 80);
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'white';
  ctx.font = `${fontSize}px "Segoe UI"`;

  // draw all detected objects
  for (const obj of res) {
    ctx.fillStyle = 'black';
    ctx.fillText(`${Math.round(100 * obj.score)}% ${obj.label}`, obj.x + 1, obj.y + 1);
    ctx.fillStyle = 'white';
    ctx.fillText(`${Math.round(100 * obj.score)}% ${obj.label}`, obj.x, obj.y);
  }
  ctx.stroke();

  const connectParts = (parts, color) => {
    ctx.strokeStyle = color;
    ctx.beginPath();
    for (let i = 0; i < parts.length; i++) {
      const part = res.find((a) => a.label === parts[i]);
      if (part) {
        if (i === 0) ctx.moveTo(part.x, part.y);
        else ctx.lineTo(part.x, part.y);
      }
    }
    ctx.stroke();
  };

  connectParts(['head', 'neck', 'chest', 'pelvis'], '#99FFFF');
  connectParts(['rightShoulder', 'rightElbow', 'rightWrist'], '#99CCFF');
  connectParts(['leftShoulder', 'leftElbow', 'leftWrist'], '#99CCFF');
  connectParts(['rightHip', 'rightKnee', 'rightAnkle'], '#9999FF');
  connectParts(['leftHip', 'leftKnee', 'leftAnkle'], '#9999FF');
  connectParts(['rightShoulder', 'leftShoulder', 'leftHip', 'rightHip', 'rightShoulder'], '#9900FF');

  // write canvas to jpeg
  const outImage = `outputs/${path.basename(img.fileName)}`;
  const out = fs.createWriteStream(outImage);
  out.on('finish', () => log.state('Created output image:', outImage, 'size:', [c.width, c.height]));
  out.on('error', (err) => log.error('Error creating image:', outImage, err));
  const stream = c.createJPEGStream({ quality: 0.6, progressive: true, chromaSubsampling: true });
  stream.pipe(out);
}

// eslint-disable-next-line no-unused-vars
function padImage(imgTensor) {
  return tf.tidy(() => {
    const [height, width] = imgTensor.shape.slice(1);
    if (height === width) return imgTensor;
    const axis = height > width ? 2 : 1;

    const createPaddingTensor = (ammount) => {
      const paddingTensorShape = imgTensor.shape.slice();
      paddingTensorShape[axis] = ammount;
      return tf.fill(paddingTensorShape, 0, 'float32');
    };

    let ammount = 0;
    const diff = Math.abs(height - width);
    ammount = Math.round(diff * 0.5);
    const append = createPaddingTensor(ammount);
    ammount = diff - ammount; // (append.shape[axis] || 0);
    const prepend = createPaddingTensor(ammount);
    return tf.concat([prepend, imgTensor, append], axis);
  });
}

// load image from file and prepares image tensor that fits the model
async function loadImage(fileName, inputSize) {
  const data = fs.readFileSync(fileName);
  const obj = tf.tidy(() => {
    const buffer = tf.node.decodeImage(data);
    const expand = buffer.expandDims(0);
    const cast = expand.cast('float32');
    // const pad = padImage(cast);
    const pad = expand;
    // @ts-ignore
    const resize = tf.image.resizeBilinear(cast, [inputSize, inputSize]);
    const normalize = resize.div(127.5).sub(1);
    const tensor = normalize;
    const img = { fileName, tensor, inputShape: buffer?.shape, paddedShape: pad?.shape, modelShape: tensor?.shape, size: buffer?.size };
    return img;
  });
  return obj;
}

// performs argmax and max functions on a 2d tensor
function max2d(inputs) {
  const [width, height] = inputs.shape;
  return tf.tidy(() => {
    // modulus op implemented in tf
    const mod = (a, b) => tf.sub(a, tf.mul(tf.div(a, tf.scalar(b, 'int32')), tf.scalar(b, 'int32')));
    // combine all data
    const reshaped = tf.reshape(inputs, [height * width]);
    // get highest score
    const score = tf.max(reshaped, 0).dataSync()[0];
    if (score > modelOptions.minScore) {
      // skip coordinate calculation is score is too low
      const coords = tf.argMax(reshaped, 0);
      const x = mod(coords, width).dataSync()[0];
      const y = tf.div(coords, tf.scalar(width, 'int32')).dataSync()[0];
      return [x, y, score];
    }
    return [0, 0, score];
  });
}

// process model results
async function processResults(res, img) {
  const squeeze = res.squeeze();
  tf.dispose(res);
  // body parts are basically just a stack of 2d tensors
  const stack = squeeze.unstack(2);
  tf.dispose(squeeze);
  const parts = [];
  // process each unstacked tensor as a separate body part
  for (let id = 0; id < stack.length; id++) {
    // actual processing to get coordinates and score
    const [x, y, score] = max2d(stack[id]);
    const [xRaw, yRaw] = [ // x, y normalized to 0..1
      x / img.modelShape[2],
      y / img.modelShape[1],
    ];
    if (score > modelOptions.minScore) {
      parts.push({
        id,
        score,
        label: bodyParts[id],
        xRaw,
        yRaw,
        x: Math.round(img.inputShape[1] * xRaw), // x normalized to input image size
        y: Math.round(img.inputShape[0] * yRaw), // y normalized to input image size
      });
    }
  }
  stack.forEach((a) => tf.dispose(a));
  return parts;
}

async function main() {
  log.header();

  // init tensorflow
  await tf.enableProdMode();
  await tf.setBackend('tensorflow');
  await tf.ENV.set('DEBUG', false);
  await tf.ready();

  // load model
  const model = await tf.loadGraphModel(modelOptions.modelPath);
  log.info('Loaded model', modelOptions, 'tensors:', tf.engine().memory().numTensors, 'bytes:', tf.engine().memory().numBytes);
  // @ts-ignore
  log.info('Model Signature', model.signature);

  // load image and get approprite tensor for it
  const inputSize = Object.values(model.modelSignature['inputs'])[0].tensorShape.dim[2].size;
  const imageFile = process.argv.length > 2 ? process.argv[2] : null;
  if (!imageFile || !fs.existsSync(imageFile)) {
    log.error('Specify a valid image file');
    process.exit();
  }
  const img = await loadImage(imageFile, inputSize);
  log.info('Loaded image:', img.fileName, 'inputShape:', img.inputShape, 'paddedShape', img.paddedShape, 'modelShape:', img.modelShape, 'decoded size:', img.size);

  // run actual prediction
  const t0 = process.hrtime.bigint();
  const res = await model.executeAsync(img.tensor);
  const t1 = process.hrtime.bigint();
  log.info('Inference time:', Math.round(parseInt((t1 - t0).toString()) / 1000 / 1000), 'ms');

  // process results
  const results = await processResults(res, img);
  const t2 = process.hrtime.bigint();
  log.info('Processing time:', Math.round(parseInt((t2 - t1).toString()) / 1000 / 1000), 'ms');

  // print results
  log.data('Results:', results);

  // save processed image
  await saveImage(results, img);
}

main();
