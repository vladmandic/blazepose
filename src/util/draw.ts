/**
 * Module that implements helper draw functions, exposed as human.draw
 */

import { mergeDeep, now } from './util';
import type { Point, Result, BodyResult, GestureResult, PersonResult } from './result';

/**
  * Draw Options
  * Accessed via `human.draw.options` or provided per each draw method as the drawOptions optional parameter
  * -color: draw color
  * -labelColor: color for labels
  * -shadowColor: optional shadow color for labels
  * -font: font for labels
  * -lineHeight: line height for labels, used for multi-line labels,
  * -lineWidth: width of any lines,
  * -pointSize: size of any point,
  * -roundRect: for boxes, round corners by this many pixels,
  * -drawPoints: should points be drawn,
  * -drawLabels: should labels be drawn,
  * -drawBoxes: should boxes be drawn,
  * -drawPolygons: should polygons be drawn,
  * -fillPolygons: should drawn polygons be filled,
  * -useDepth: use z-axis coordinate as color shade,
  * -useCurves: draw polygons as cures or as lines,
  * -bufferedOutput: experimental: allows to call draw methods multiple times for each detection and interpolate results between results thus achieving smoother animations
  */
export interface DrawOptions {
   color: string,
   labelColor: string,
   shadowColor: string,
   font: string,
   lineHeight: number,
   lineWidth: number,
   pointSize: number,
   roundRect: number,
   drawPoints: boolean,
   drawLabels: boolean,
   drawBoxes: boolean,
   drawPolygons: boolean,
   drawGaze: boolean,
   fillPolygons: boolean,
   useDepth: boolean,
   useCurves: boolean,
   bufferedOutput: boolean,
 }

export const options: DrawOptions = {
  color: <string>'rgba(173, 216, 230, 0.6)', // 'lightblue' with light alpha channel
  labelColor: <string>'rgba(173, 216, 230, 1)', // 'lightblue' with dark alpha channel
  shadowColor: <string>'black',
  font: <string>'small-caps 18px "Segoe UI"',
  lineHeight: <number>18,
  lineWidth: <number>4,
  pointSize: <number>2,
  roundRect: <number>8,
  drawPoints: <boolean>true,
  drawLabels: <boolean>true,
  drawBoxes: <boolean>true,
  drawPolygons: <boolean>true,
  drawGaze: <boolean>true,
  fillPolygons: <boolean>false,
  useDepth: <boolean>true,
  useCurves: <boolean>false,
  bufferedOutput: <boolean>true,
};

const getCanvasContext = (input) => {
  if (input && input.getContext) return input.getContext('2d');
  throw new Error('invalid canvas');
};

function point(ctx, x, y, z = 0, localOptions) {
  ctx.fillStyle = localOptions.useDepth && z ? `rgba(${127.5 + (2 * z)}, ${127.5 - (2 * z)}, 255, 0.3)` : localOptions.color;
  ctx.beginPath();
  ctx.arc(x, y, localOptions.pointSize, 0, 2 * Math.PI);
  ctx.fill();
}

function rect(ctx, x, y, width, height, localOptions) {
  ctx.beginPath();
  if (localOptions.useCurves) {
    const cx = (x + x + width) / 2;
    const cy = (y + y + height) / 2;
    ctx.ellipse(cx, cy, width / 2, height / 2, 0, 0, 2 * Math.PI);
  } else {
    ctx.lineWidth = localOptions.lineWidth;
    ctx.moveTo(x + localOptions.roundRect, y);
    ctx.lineTo(x + width - localOptions.roundRect, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + localOptions.roundRect);
    ctx.lineTo(x + width, y + height - localOptions.roundRect);
    ctx.quadraticCurveTo(x + width, y + height, x + width - localOptions.roundRect, y + height);
    ctx.lineTo(x + localOptions.roundRect, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - localOptions.roundRect);
    ctx.lineTo(x, y + localOptions.roundRect);
    ctx.quadraticCurveTo(x, y, x + localOptions.roundRect, y);
    ctx.closePath();
  }
  ctx.stroke();
}

function lines(ctx, points: Array<Point> = [], localOptions) {
  if (points === undefined || points.length === 0) return;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (const pt of points) {
    const z = pt[2] || 0;
    ctx.strokeStyle = localOptions.useDepth && z ? `rgba(${127.5 + (2 * z)}, ${127.5 - (2 * z)}, 255, 0.3)` : localOptions.color;
    ctx.fillStyle = localOptions.useDepth && z ? `rgba(${127.5 + (2 * z)}, ${127.5 - (2 * z)}, 255, 0.3)` : localOptions.color;
    ctx.lineTo(pt[0], Math.round(pt[1]));
  }
  ctx.stroke();
  if (localOptions.fillPolygons) {
    ctx.closePath();
    ctx.fill();
  }
}

function curves(ctx, points: [number, number, number?][] = [], localOptions) {
  if (points === undefined || points.length === 0) return;
  if (!localOptions.useCurves || points.length <= 2) {
    lines(ctx, points, localOptions);
    return;
  }
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 0; i < points.length - 2; i++) {
    const xc = (points[i][0] + points[i + 1][0]) / 2;
    const yc = (points[i][1] + points[i + 1][1]) / 2;
    ctx.quadraticCurveTo(points[i][0], points[i][1], xc, yc);
  }
  ctx.quadraticCurveTo(points[points.length - 2][0], points[points.length - 2][1], points[points.length - 1][0], points[points.length - 1][1]);
  ctx.stroke();
  if (localOptions.fillPolygons) {
    ctx.closePath();
    ctx.fill();
  }
}

export async function gesture(inCanvas: HTMLCanvasElement | OffscreenCanvas, result: Array<GestureResult>, drawOptions?: Partial<DrawOptions>) {
  const localOptions = mergeDeep(options, drawOptions);
  if (!result || !inCanvas) return;
  const ctx = getCanvasContext(inCanvas);
  ctx.font = localOptions.font;
  ctx.fillStyle = localOptions.color;
  let i = 1;
  for (let j = 0; j < result.length; j++) {
    let where: unknown[] = []; // what&where is a record
    let what: unknown[] = []; // what&where is a record
    [where, what] = Object.entries(result[j]);
    if ((what.length > 1) && ((what[1] as string).length > 0)) {
      const who = where[1] as number > 0 ? `#${where[1]}` : '';
      const label = `${where[0]} ${who}: ${what[1]}`;
      if (localOptions.shadowColor && localOptions.shadowColor !== '') {
        ctx.fillStyle = localOptions.shadowColor;
        ctx.fillText(label, 8, 2 + (i * localOptions.lineHeight));
      }
      ctx.fillStyle = localOptions.labelColor;
      ctx.fillText(label, 6, 0 + (i * localOptions.lineHeight));
      i += 1;
    }
  }
}

export async function body(inCanvas: HTMLCanvasElement | OffscreenCanvas, result: Array<BodyResult>, drawOptions?: Partial<DrawOptions>) {
  const localOptions = mergeDeep(options, drawOptions);
  if (!result || !inCanvas) return;
  const ctx = getCanvasContext(inCanvas);
  ctx.lineJoin = 'round';
  for (let i = 0; i < result.length; i++) {
    ctx.strokeStyle = localOptions.color;
    ctx.fillStyle = localOptions.color;
    ctx.lineWidth = localOptions.lineWidth;
    ctx.font = localOptions.font;
    if (localOptions.drawBoxes && result[i].box && result[i].box?.length === 4) {
      rect(ctx, result[i].box[0], result[i].box[1], result[i].box[2], result[i].box[3], localOptions);
      if (localOptions.drawLabels) {
        if (localOptions.shadowColor && localOptions.shadowColor !== '') {
          ctx.fillStyle = localOptions.shadowColor;
          ctx.fillText(`body ${100 * result[i].score}%`, result[i].box[0] + 3, 1 + result[i].box[1] + localOptions.lineHeight, result[i].box[2]);
        }
        ctx.fillStyle = localOptions.labelColor;
        ctx.fillText(`body ${100 * result[i].score}%`, result[i].box[0] + 2, 0 + result[i].box[1] + localOptions.lineHeight, result[i].box[2]);
      }
    }
    if (localOptions.drawPoints) {
      for (let pt = 0; pt < result[i].keypoints.length; pt++) {
        ctx.fillStyle = localOptions.useDepth && result[i].keypoints[pt].position[2] ? `rgba(${127.5 + (2 * (result[i].keypoints[pt].position[2] || 0))}, ${127.5 - (2 * (result[i].keypoints[pt].position[2] || 0))}, 255, 0.5)` : localOptions.color;
        point(ctx, result[i].keypoints[pt].position[0], result[i].keypoints[pt].position[1], 0, localOptions);
      }
    }
    if (localOptions.drawLabels) {
      ctx.font = localOptions.font;
      if (result[i].keypoints) {
        for (const pt of result[i].keypoints) {
          ctx.fillStyle = localOptions.useDepth && pt.position[2] ? `rgba(${127.5 + (2 * pt.position[2])}, ${127.5 - (2 * pt.position[2])}, 255, 0.5)` : localOptions.color;
          ctx.fillText(`${pt.part} ${Math.trunc(100 * pt.score)}%`, pt.position[0] + 4, pt.position[1] + 4);
        }
      }
    }
    if (localOptions.drawPolygons && result[i].keypoints && result[i].annotations) {
      for (const part of Object.values(result[i].annotations)) {
        for (const connected of part) curves(ctx, connected, localOptions);
      }
    }
  }
}

export async function person(inCanvas: HTMLCanvasElement | OffscreenCanvas, result: Array<PersonResult>, drawOptions?: Partial<DrawOptions>) {
  const localOptions = mergeDeep(options, drawOptions);
  if (!result || !inCanvas) return;
  const ctx = getCanvasContext(inCanvas);
  ctx.lineJoin = 'round';
  ctx.font = localOptions.font;

  for (let i = 0; i < result.length; i++) {
    if (localOptions.drawBoxes) {
      ctx.strokeStyle = localOptions.color;
      ctx.fillStyle = localOptions.color;
      rect(ctx, result[i].box[0], result[i].box[1], result[i].box[2], result[i].box[3], localOptions);
      if (localOptions.drawLabels) {
        const label = `person #${i}`;
        if (localOptions.shadowColor && localOptions.shadowColor !== '') {
          ctx.fillStyle = localOptions.shadowColor;
          ctx.fillText(label, result[i].box[0] + 3, 1 + result[i].box[1] + localOptions.lineHeight, result[i].box[2]);
        }
        ctx.fillStyle = localOptions.labelColor;
        ctx.fillText(label, result[i].box[0] + 2, 0 + result[i].box[1] + localOptions.lineHeight, result[i].box[2]);
      }
      ctx.stroke();
    }
  }
}

export async function canvas(input: HTMLCanvasElement | OffscreenCanvas | HTMLImageElement | HTMLMediaElement | HTMLVideoElement, output: HTMLCanvasElement) {
  if (!input || !output) return;
  const ctx = getCanvasContext(output);
  ctx.drawImage(input, 0, 0);
}

export async function all(inCanvas: HTMLCanvasElement | OffscreenCanvas, result: Result, drawOptions?: Partial<DrawOptions>) {
  if (!result || !result || !inCanvas) return null;
  const timestamp = now();
  const localOptions = mergeDeep(options, drawOptions);
  const promise = Promise.all([
    body(inCanvas, result.body, localOptions),
    gesture(inCanvas, result.gesture, localOptions), // gestures do not have buffering
    // person(inCanvas, result.persons, localOptions); // already included above
  ]);
  if (result.performance) result.performance.draw = Math.trunc(now() - timestamp);
  return promise;
}
