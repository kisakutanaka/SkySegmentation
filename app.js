import { createSkySegmenter } from './sky-segmenter.js';

const SMILEY = '😀';
const SMILEY_SCALE = 0.28; // 画面短辺に対するスマイリーの大きさ
const MASK_INERTIA = 0.6; // マスクの時間平滑化（大きいほどブレないが追従が遅い）

const video = document.getElementById('camera');
const view = document.getElementById('view');
const viewCtx = view.getContext('2d');
const startBtn = document.getElementById('start');
const statusEl = document.getElementById('status');
const statsEl = document.getElementById('stats');

// 前景（空以外）だけを切り抜くための作業用キャンバス
const fg = document.createElement('canvas');
const fgCtx = fg.getContext('2d');

// 低解像度マスク。アルファチャンネルに「前景である度合い」を書き込む
const maskCanvas = document.createElement('canvas');
const maskCtx = maskCanvas.getContext('2d');
let maskImage = null;
let smoothed = null; // 時間平滑化した空確率
const skyCenter = { x: 0.5, y: 0.3 }; // 空領域の重心（スマイリーの位置）

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
    const segmenter = await createSkySegmenter();

    statusEl.textContent = '';
    startBtn.hidden = true;
    statsEl.hidden = false;
    running = true;

    requestAnimationFrame(render); // 描画は 60fps 側
    inferenceLoop(segmenter); // 推論は端末性能なりの速度で回す
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
async function inferenceLoop(segmenter) {
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
    maskCanvas.width = width;
    maskCanvas.height = height;
    maskImage = maskCtx.createImageData(width, height);
  } else {
    for (let i = 0; i < data.length; i++) {
      smoothed[i] = smoothed[i] * MASK_INERTIA + data[i] * (1 - MASK_INERTIA);
    }
  }

  // アルファ = 1 - 空確率 = 前景として残す度合い。あわせて空の重心も求める。
  const px = maskImage.data;
  let sumX = 0;
  let sumY = 0;
  let sumSky = 0;
  for (let y = 0, i = 0; y < height; y++) {
    for (let x = 0; x < width; x++, i++) {
      const sky = smoothed[i];
      px[i * 4 + 3] = (1 - sky) * 255;
      sumX += x * sky;
      sumY += y * sky;
      sumSky += sky;
    }
  }
  maskCtx.putImageData(maskImage, 0, 0);

  if (sumSky > width * height * 0.02) {
    skyCenter.x += (sumX / sumSky / width - skyCenter.x) * 0.2;
    skyCenter.y += (sumY / sumSky / height - skyCenter.y) * 0.2;
  }
}

function render(now) {
  if (!running) return;
  requestAnimationFrame(render);

  const w = view.width;
  const h = view.height;

  // 1. 後景：カメラ映像をそのまま
  viewCtx.drawImage(video, 0, 0, w, h);

  // 2. 空に浮かぶスマイリー
  const size = Math.min(w, h) * SMILEY_SCALE;
  viewCtx.font = `${size}px "Apple Color Emoji", "Noto Color Emoji", "Segoe UI Emoji", sans-serif`;
  viewCtx.textAlign = 'center';
  viewCtx.textBaseline = 'middle';
  viewCtx.fillText(SMILEY, skyCenter.x * w, skyCenter.y * h + Math.sin(now / 900) * size * 0.06);

  // 3. 前景（空以外）だけを切り抜く：マスクのアルファで destination-in
  if (maskImage) {
    fgCtx.globalCompositeOperation = 'source-over';
    fgCtx.clearRect(0, 0, w, h);
    fgCtx.drawImage(video, 0, 0, w, h);
    fgCtx.globalCompositeOperation = 'destination-in';
    fgCtx.drawImage(maskCanvas, 0, 0, w, h); // 低解像度→全画面の拡大がそのまま境界のぼかしになる
    // 4. 前景を最前面へ。これでスマイリーが建物や人の背後に回り込む
    viewCtx.drawImage(fg, 0, 0);
  }

  if (lastFrame) fps = fps * 0.9 + (1000 / (now - lastFrame)) * 0.1;
  lastFrame = now;
  statsEl.textContent = `推論 ${inferMs.toFixed(0)} ms / 描画 ${fps.toFixed(0)} fps`;
}
