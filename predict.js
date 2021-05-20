const fs = require('fs');
const path = require('path');
const process = require('process');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');
const canvas = require('canvas');
const blazepose = require('./src/blazepose');

// save image with processed results
async function saveImage(results, img) {
  // load and draw original image
  const original = await canvas.loadImage(img);
  // create canvas
  const c = new canvas.Canvas(original.width, original.height);
  const ctx = c.getContext('2d');
  ctx.drawImage(original, 0, 0, c.width, c.height);
  // const fontSize = Math.trunc(c.width / 50);
  const fontSize = Math.round((c.width * c.height) ** (1 / 2) / 50);
  ctx.lineWidth = 2;
  ctx.font = `${fontSize}px "Segoe UI"`;

  // draw all detected objects
  for (const res of results) {
    // eslint-disable-next-line no-continue
    if (!res.keypoints) continue;

    let color = 'white';
    if (res.name === 'detect') color = 'lightcoral';
    if (res.name === 'full') color = 'lightblue';
    if (res.name === 'upper') color = 'lightgreen';
    ctx.strokeStyle = color;
    for (const pt of res.keypoints) {
      ctx.fillStyle = 'black';
      ctx.fillText(`${Math.round(100 * (pt.score || 0))}% ${pt.part || pt.id}`, pt.position.x + 1, pt.position.y + 1);
      ctx.fillStyle = color;
      ctx.fillText(`${Math.round(100 * (pt.score || 0))}% ${pt.part || pt.id}`, pt.position.x, pt.position.y);
    }
    ctx.stroke();

    const connectParts = (parts) => {
      ctx.beginPath();
      for (let i = 0; i < parts.length; i++) {
        const part = res.keypoints.find((a) => a.part === parts[i]);
        if (part) {
          if (i === 0) ctx.moveTo(part.position.x, part.position.y);
          else ctx.lineTo(part.position.x, part.position.y);
        }
      }
      ctx.stroke();
    };

    connectParts(['leftEar', 'leftEyeOutside', 'leftEye', 'leftEyeInside', 'nose', 'leftMouth', 'rightMouth', 'nose', 'forehead', 'nose', 'rightEyeInside', 'rightEye', 'rightEyeOutside', 'rightEar']);
    connectParts(['leftPalm', 'leftWrist', 'leftElbow', 'leftShoulder']);
    connectParts(['rightPalm', 'rightWrist', 'rightElbow', 'rightShoulder']);
    connectParts(['leftHip', 'leftKnee', 'leftAnkle', 'leftHeel', 'leftFoot']);
    connectParts(['rightHip', 'rightKnee', 'rightAnkle', 'rightHeel', 'rightFoot']);
    connectParts(['leftHip', 'midHip', 'rightHip', 'rightShoulder', 'leftShoulder', 'leftHip']);
  }

  // write canvas to jpeg
  const outImage = `outputs/${path.basename(img)}`;
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
async function loadImage(fileName) {
  const data = fs.readFileSync(fileName);
  const obj = tf.tidy(() => {
    const buffer = tf.node.decodeImage(data);
    const cast = buffer.cast('float32');
    const expand = cast.expandDims(0);
    // const pad = padImage(expand);
    // @ts-ignore
    return expand;
  });
  return obj;
}

async function main() {
  log.header();

  // init tensorflow
  await tf.enableProdMode();
  await tf.setBackend('tensorflow');
  await tf.ENV.set('DEBUG', false);
  await tf.ready();

  // load model
  const models = await blazepose.load();
  for (const model of models) {
    log.info('Loaded model', model.modelUrl, 'tensors:', tf.engine().memory().numTensors, 'bytes:', tf.engine().memory().numBytes);
    log.info('Model Signature', model.signature);
  }
  log.info('TFJS', 'tensors:', tf.engine().memory().numTensors, 'bytes:', tf.engine().memory().numBytes);

  const imageFile = process.argv.length > 2 ? process.argv[2] : null;
  if (!imageFile || !fs.existsSync(imageFile)) {
    log.error('Specify a valid image file');
    process.exit();
  }
  const img = await loadImage(imageFile);
  log.info('Loaded image:', imageFile, 'inputShape:', img.shape, 'decoded size:', img.size);

  // run actual prediction
  const t0 = process.hrtime.bigint();
  const res = await blazepose.predict(img, { detect: true, full: true, upper: true });
  tf.dispose(img);
  const t1 = process.hrtime.bigint();
  log.info('Inference time:', Math.round(parseInt((t1 - t0).toString()) / 1000 / 1000), 'ms');

  // print results
  log.data('Results:', res);

  // save processed image
  await saveImage(res, imageFile);
}

main();
