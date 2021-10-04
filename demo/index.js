import * as blazepose from '../dist/blazepose.js';

const config = {
  modelBasePath: 'model',
  cacheSensitivity: 0.75,
  skipFrame: true,
  body: {
    enabled: true,
    modelPath: 'blazepose-lite.json',
    detector: { modelPath: '' },
    maxDetected: 1,
    minConfidence: 0.55,
    skipFrames: 1,
  },
};

// eslint-disable-next-line no-console
const log = (...msg) => console.log(...msg);

/** @type {HTMLVideoElement} */ // @ts-ignore
const video = document.getElementById('video') || document.createElement('video'); // used as input
/** @type {HTMLCanvasElement} */ // @ts-ignore
const canvas = document.getElementById('canvas') || document.createElement('canvas'); // used as output
const fps = { detect: 0, draw: 0 };
fps.el = document.getElementById('fps') || document.createElement('div'); // used as draw fps counter

async function initWebCam() {
  const constraints = { audio: false, video: { facingMode: 'user', resizeMode: 'none', width: { ideal: document.body.clientWidth } } }; // set preffered camera options
  const stream = await navigator.mediaDevices.getUserMedia(constraints); // get webcam stream that matches constraints
  const ready = new Promise((resolve) => { video.onloadeddata = () => resolve(true); }); // resolve when stream is ready
  video.srcObject = stream; // assign stream to video element
  video.play(); // start stream
  await ready; // wait until stream is ready
  canvas.width = video.videoWidth; // resize output canvas to match input
  canvas.height = video.videoHeight;
  log('video stream:', video.srcObject, 'track state:', video.srcObject.getVideoTracks()[0].readyState, 'stream state:', video.readyState);
  canvas.onclick = () => { // play or pause on mouse click
    if (video.paused) video.play();
    else video.pause();
  };
}

async function detectionLoop() {
  const t0 = performance.now();
  if (!video.paused) {
    const tensor = blazepose.util.input2tensor(video);
    const body = await blazepose.predict(tensor, config);
    blazepose.util.dispose(tensor);
    // eslint-disable-next-line no-console
    if (body && body[0]) console.log('BODY:', body[0]);
    const result = { body };
    await blazepose.draw.canvas(video, canvas); // draw input video to output canvas
    await blazepose.draw.all(canvas, result); // draw results as overlay on output canvas
  }
  const t1 = performance.now();
  fps.detect = 1000 / (t1 - t0);
  fps.el.innerText = video.paused ? 'paused' : `FPS: ${fps.detect.toFixed(1)} Time: ${(t1 - t0).toFixed(0)} ms`;
  requestAnimationFrame(detectionLoop); // run in loop
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
async function detectImage(url) {
  const image = document.createElement('img');
  const loaded = new Promise((resolve) => image.onload = () => resolve(true));
  image.src = url;
  await loaded;
  const tensor = blazepose.util.input2tensor(image);
  const body = await blazepose.predict(tensor, config);
  blazepose.util.dispose(tensor);
  // eslint-disable-next-line no-console
  if (body && body[0]) console.log('BODY:', body[0]);
  const result = { body };
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  await blazepose.draw.canvas(image, canvas); // draw input video to output canvas
  await blazepose.draw.all(canvas, result); // draw results as overlay on output canvas
}

async function main() {
  await blazepose.load(config);
  await initWebCam();
  await detectionLoop();
  // await detectImage('daz3d-ella.jpg');
}

window.onload = main;
