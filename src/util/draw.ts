import type { Point } from './types';

export const options = {
  color: <string>'rgba(173, 216, 230, 0.6)', // 'lightblue' with light alpha channel
  labelColor: <string>'rgba(173, 216, 230, 1)', // 'lightblue' with dark alpha channel
  font: <string>'small-caps 22px "Segoe UI"',
  lineWidth: <number>8,
  pointSize: <number>8,
  roundRect: <number>8,
};

const getCanvasContext = (input) => {
  if (input && input.getContext) return input.getContext('2d');
  throw new Error('invalid canvas');
};

function point(ctx, x, y, z) {
  ctx.fillStyle = `rgba(${127.5 + (2 * z)}, ${127.5 - (2 * z)}, 255, 0.3)`;
  ctx.beginPath();
  ctx.arc(x, y, options.pointSize, 0, 2 * Math.PI);
  ctx.fill();
}

function rect(ctx, x, y, width, height) {
  ctx.beginPath();
  ctx.lineWidth = options.lineWidth;
  ctx.moveTo(x, y);
  ctx.lineTo(x + width, y);
  ctx.lineTo(x + width, y + height);
  ctx.lineTo(x, y + height);
  ctx.lineTo(x, y);
  ctx.closePath();
  ctx.stroke();
}

function lines(ctx, points: Array<Point>) {
  if (points === undefined || points.length === 0) return;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (const pt of points) {
    const z = pt[2] || 0;
    ctx.strokeStyle = `rgba(${127.5 + (2 * z)}, ${127.5 - (2 * z)}, 255, 0.3)`;
    ctx.lineTo(pt[0], Math.round(pt[1]));
  }
  ctx.stroke();
}

export async function body(inCanvas: HTMLCanvasElement | OffscreenCanvas, result) {
  if (!result || !inCanvas) return;
  const ctx = getCanvasContext(inCanvas);
  ctx.lineJoin = 'round';
  for (let i = 0; i < result.length; i++) {
    ctx.strokeStyle = options.color;
    ctx.fillStyle = options.color;
    ctx.lineWidth = options.lineWidth;
    ctx.font = options.font;
    rect(ctx, result[i].box[0], result[i].box[1], result[i].box[2], result[i].box[3]);
    ctx.fillStyle = options.labelColor;
    ctx.fillText(`body ${100 * result[i].score}%`, result[i].box[0] + 6, 0 + result[i].box[1] + 22, result[i].box[2]);
    for (let pt = 0; pt < result[i].keypoints.length; pt++) {
      ctx.fillStyle = `rgba(${127.5 + (2 * (result[i].keypoints[pt].position[2] || 0))}, ${127.5 - (2 * (result[i].keypoints[pt].position[2] || 0))}, 255, 0.5)`;
      point(ctx, result[i].keypoints[pt].position[0], result[i].keypoints[pt].position[1], 0);
    }
    ctx.font = options.font;
    if (result[i].keypoints) {
      for (const pt of result[i].keypoints) {
        ctx.fillStyle = `rgba(${127.5 + (2 * pt.position[2])}, ${127.5 - (2 * pt.position[2])}, 255, 0.5)`;
        ctx.fillText(`${pt.part} ${pt.score} ${pt.presence}`, pt.position[0] + 8, pt.position[1] + 6);
      }
    }
    for (const part of Object.values(result[i].annotations)) {
      // @ts-ignore
      for (const connected of part) lines(ctx, connected);
    }
  }
}

export async function canvas(input: HTMLCanvasElement | OffscreenCanvas | HTMLImageElement | HTMLMediaElement | HTMLVideoElement, output: HTMLCanvasElement) {
  if (!input || !output) return;
  const ctx = getCanvasContext(output);
  ctx.drawImage(input, 0, 0);
}
