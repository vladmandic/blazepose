const fs = require('fs');
const zlib = require('zlib');
const http = require('http');
const path = require('path');
// eslint-disable-next-line node/no-unpublished-require, import/no-extraneous-dependencies
const log = require('@vladmandic/pilogger');

const httpPort = 8000;

// just some predefined mime types
const mime = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpg',
  '.gif': 'image/gif',
  '.ico': 'image/x-icon',
  '.svg': 'image/svg+xml',
  '.wav': 'audio/wav',
  '.mp4': 'video/mp4',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.wasm': 'application/wasm',
};

// get file content for a valid url request
function handle(url) {
  console.log(url);
  return new Promise((resolve) => {
    const obj = { ok: false };
    if (fs.existsSync(url)) obj.file = url;
    if (!obj.file) resolve(null);
    obj.stat = fs.statSync(obj.file);
    if (obj.stat.isFile()) obj.ok = true;
    resolve(obj);
  });
}

// process http requests
async function httpRequest(req, res) {
  handle(path.join(__dirname, decodeURI(req.url)))
    .then((result) => {
      // get original ip of requestor, regardless if it's behind proxy or not
      // eslint-disable-next-line dot-notation
      const forwarded = (req.headers['forwarded'] || '').match(/for="\[(.*)\]:/);
      const ip = (Array.isArray(forwarded) ? forwarded[1] : null) || req.headers['x-forwarded-for'] || req.ip || req.socket.remoteAddress;
      if (!result || !result.ok) {
        res.writeHead(404, { 'Content-Type': 'text/html' });
        res.end('Error 404: Not Found\n', 'utf-8');
        log.warn(`${req.method}/${req.httpVersion}`, res.statusCode, req.url, ip);
      } else {
        const ext = String(path.extname(result.file)).toLowerCase();
        const contentType = mime[ext] || 'application/octet-stream';
        const accept = req.headers['accept-encoding'] ? req.headers['accept-encoding'].includes('br') : false; // does target accept brotli compressed data
        res.writeHead(200, {
          // 'Content-Length': result.stat.size, // not using as it's misleading for compressed streams
          'Content-Language': 'en', 'Content-Type': contentType, 'Content-Encoding': accept ? 'br' : '', 'Last-Modified': result.stat.mtime, 'Cache-Control': 'no-cache', 'X-Content-Type-Options': 'nosniff',
        });
        const compress = zlib.createBrotliCompress({ params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 5 } }); // instance of brotli compression with level 5
        const stream = fs.createReadStream(result.file);
        if (!accept) stream.pipe(res); // don't compress data
        else stream.pipe(compress).pipe(res); // compress data
        log.data(`${req.method}/${req.httpVersion}`, res.statusCode, contentType, result.stat.size, req.url, ip);
      }
      return null;
    })
    .catch((err) => log.error('handle error:', err));
}

// app main entry point
async function main() {
  log.header();
  const server1 = http.createServer({}, httpRequest);
  server1.on('listening', () => log.state('HTTP server listening:', httpPort));
  server1.listen(httpPort);
}

main();
