import { createSkySegmenter } from './sky-segmenter.js';

const KAIJU_SRC = './kaiju.png';
const KAIJU_SCALE = 0.45; // 画面短辺に対する怪獣の大きさ
const KAIJU_POS = { x: 0.5, y: 0.5 }; // 画面内の位置（0〜1）
const MASK_INERTIA = 0.6; // マスクの時間平滑化（大きいほどブレないが追従が遅い）

const video = document.getElementById('camera');
const view = document.getElementById('view');
const viewCtx = view.getContext('2d');
const startBtn = document.getElementById('start');
const statusEl = document.getElementById('status');
const statsEl = document.getElementById('stats');
const debugEl = document.getElementById('debug');

// 前景（空以外）だけを切り抜くための作業用キャンバス
const fg = document.createElement('canvas');
const fgCtx = fg.getContext('2d');

// 低解像度マスク。アルファチャンネルに「前景である度合い」を書き込む
const maskCanvas = document.createElement('canvas');
const maskCtx = maskCanvas.getContext('2d');
let maskImage = null;
// デバッグ表示用。こちらは逆に「空である度合い」を色付きで持つ
const skyCanvas = document.createElement('canvas');
const skyCtx = skyCanvas.getContext('2d');
let skyImage = null;
let smoothed = null; // 時間平滑化した空確率

const kaiju = new Image();
kaiju.src = KAIJU_SRC;

let segmenter = null;
let running = false;
let inferMs = 0;
let fps = 0;
let lastFrame = 0;

startBtn.addEventListener('click', start, { once: true });

async function start() {
  startBtn.disabled = true;
  try {
    statusEl.textContent = 'カメラを起動しています…';
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    view.width = fg.width = video.videoWidth;
    view.height = fg.height = video.videoHeight;

    statusEl.textContent = 'モデルを読み込んでいます…';
    segmenter = await createSkySegmenter();

    statusEl.textContent = '';
    startBtn.hidden = true;
    statsEl.hidden = false;
    running = true;

    requestAnimationFrame(render); // 描画は 60fps 側
    inferenceLoop(); // 推論は端末性能なりの速度で回す
  } catch (err) {
    console.error(err);
    statusEl.textContent = `起動できませんでした: ${err.message}`;
    startBtn.disabled = false;
  }
}

/**
 * 推論ループ。描画ループとは独立して回り、最新のマスクだけを更新する。
 * 推論が数fpsでも、描画は常に最新マスクを使い回すので映像はカクつかない。
 */
async function inferenceLoop() {
  while (running) {
    const t0 = performance.now();
    try {
      updateMask(await segmenter.segment(video));
    } catch (err) {
      console.error(err);
      statusEl.textContent = `推論エラー: ${err.message}`;
      running = false;
      return;
    }
    inferMs = performance.now() - t0;
  }
}

function updateMask({ width, height, data }) {
  if (!smoothed || smoothed.length !== data.length) {
    smoothed = Float32Array.from(data);
    maskCanvas.width = skyCanvas.width = width;
    maskCanvas.height = skyCanvas.height = height;
    maskImage = maskCtx.createImageData(width, height);
    skyImage = skyCtx.createImageData(width, height);
  } else {
    for (let i = 0; i < data.length; i++) {
      smoothed[i] = smoothed[i] * MASK_INERTIA + data[i] * (1 - MASK_INERTIA);
    }
  }

  // アルファ = 1 - 空確率 = 前景として残す度合い
  const px = maskImage.data;
  const skyPx = skyImage.data;
  for (let i = 0; i < smoothed.length; i++) {
    const sky = smoothed[i];
    px[i * 4 + 3] = (1 - sky) * 255;
    skyPx[i * 4] = 0;
    skyPx[i * 4 + 1] = 229;
    skyPx[i * 4 + 2] = 255;
    skyPx[i * 4 + 3] = sky * 255;
  }
  maskCtx.putImageData(maskImage, 0, 0);
  skyCtx.putImageData(skyImage, 0, 0);
}

function render(now) {
  if (!running) return;
  requestAnimationFrame(render);

  const w = view.width;
  const h = view.height;

  // 1. 後景：カメラ映像をそのまま
  viewCtx.drawImage(video, 0, 0, w, h);

  // デバッグ表示：空と判定された領域を塗りつぶす（合成はしない）
  if (debugEl.checked) {
    if (skyImage) {
      viewCtx.globalAlpha = 0.55;
      viewCtx.drawImage(skyCanvas, 0, 0, w, h);
      viewCtx.globalAlpha = 1;
    }
    updateStats(now);
    return;
  }

  // 2. 画面中央に怪獣
  if (kaiju.complete && kaiju.naturalWidth) {
    const kw = Math.min(w, h) * KAIJU_SCALE;
    const kh = (kw * kaiju.naturalHeight) / kaiju.naturalWidth;
    const bob = Math.sin(now / 900) * kh * 0.02;
    viewCtx.drawImage(kaiju, KAIJU_POS.x * w - kw / 2, KAIJU_POS.y * h - kh / 2 + bob, kw, kh);
  }

  // 3. 前景（空以外）だけを切り抜く：マスクのアルファで destination-in
  if (maskImage) {
    fgCtx.globalCompositeOperation = 'source-over';
    fgCtx.clearRect(0, 0, w, h);
    fgCtx.drawImage(video, 0, 0, w, h);
    fgCtx.globalCompositeOperation = 'destination-in';
    fgCtx.drawImage(maskCanvas, 0, 0, w, h); // 低解像度→全画面の拡大がそのまま境界のぼかしになる
    // 4. 前景を最前面へ。これで怪獣が建物の背後に回り込む
    viewCtx.drawImage(fg, 0, 0);
  }

  updateStats(now);
}

function updateStats(now) {
  if (lastFrame) fps = fps * 0.9 + (1000 / (now - lastFrame)) * 0.1;
  lastFrame = now;
  statsEl.textContent = `推論 ${inferMs.toFixed(0)} ms / 描画 ${fps.toFixed(0)} fps`;
}
