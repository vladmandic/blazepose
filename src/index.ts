import * as tf from '@tensorflow/tfjs';
import * as blazepose from './blazepose';
import { log } from './util/util';
import * as draw from './util/draw';

const config = {
  testImage: './daz3d-zoe.jpg',
  modelBasePath: 'model',
  body: {
    enabled: true,
    modelPath: 'blazepose-lite.json',
    detector: { enabled: false, modelPath: 'blazepose-detector.json' },
    minConfidence: 0.3,
    // maxDetected: 5,
    // iouThreshold: 0.5,
  },
};

const video = document.getElementById('video') as HTMLVideoElement;
const image = document.getElementById('image') as HTMLImageElement;
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const fps = { detect: 0, draw: 0, el: document.getElementById('fps') as HTMLDivElement };

function input2tensor(input) {
  const inputT = tf.browser.fromPixels(input);
  const expandT = tf.expandDims(inputT, 0);
  const normalizeT = tf.div(expandT, 255);
  tf.dispose([inputT, expandT]);
  return normalizeT;
}

async function initWebCam() {
  const constraints = { audio: false, video: { facingMode: 'user', resizeMode: 'none', width: { ideal: 1920 } } }; // set preffered camera options
  const stream = await navigator.mediaDevices.getUserMedia(constraints) as MediaStream; // get webcam stream that matches constraints
  const ready = new Promise((resolve) => { video.onloadeddata = () => resolve(true); }); // resolve when stream is ready
  video.srcObject = stream; // assign stream to video element
  video.play(); // start stream
  await ready; // wait until stream is ready
  canvas.width = video.videoWidth; // resize output canvas to match input
  canvas.height = video.videoHeight;
  log('video', { stream: video.srcObject, track: video.srcObject.getVideoTracks()[0], state: video.readyState });
  canvas.onclick = () => { // play or pause on mouse click
    if (video.paused) video.play();
    else video.pause();
  };
}

export async function processVideo() {
  const t0 = performance.now();
  if (!video.paused) {
    const tensor = input2tensor(video);
    const body = await blazepose.predict(tensor, config);
    tf.dispose(tensor);
    await draw.canvas(video, canvas); // draw input video to output canvas
    await draw.body(canvas, body); // draw results as overlay on output canvas
  }
  const t1 = performance.now();
  fps.detect = 1000 / (t1 - t0);
  fps.el.innerText = video.paused ? 'paused' : `FPS ${fps.detect.toFixed(1)} | Time ${(t1 - t0).toFixed(0)} ms | Tensors ${tf.engine().state.numTensors}`;
  requestAnimationFrame(processVideo); // run in loop
}

export async function processImage() {
  const ready = new Promise((resolve) => { image.onload = () => resolve(true); }); // resolve when stream is ready
  image.src = config.testImage;
  await ready;
  const t0 = performance.now();
  const tensor = input2tensor(image);
  const body = await blazepose.predict(tensor, config);
  tf.dispose(tensor);
  await draw.canvas(image, canvas); // draw input video to output canvas
  await draw.body(canvas, body); // draw results as overlay on output canvas
  const t1 = performance.now();
  fps.detect = 1000 / (t1 - t0);
  fps.el.innerText = video.paused ? 'paused' : `FPS ${fps.detect.toFixed(1)} | Time ${(t1 - t0).toFixed(0)} ms | Tensors ${tf.engine().state.numTensors}`;
}

async function main() {
  tf.env().set('WEBGL_USE_SHAPES_UNIFORMS', true);
  tf.env().set('WEBGL_EXP_CONV', true);
  tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
  await tf.ready();
  await blazepose.load(config);
  await initWebCam();
  await processVideo();
  // await processImage();
}

window.onload = main;
