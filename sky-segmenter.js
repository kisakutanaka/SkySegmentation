/**
 * sky-segmenter.js
 *
 * カメラ画像 / 画像から「空である確率マップ」だけを返す最小モジュール。
 * 依存はグローバルの `ort` (onnxruntime-web) だけなので、このファイルを
 * そのままコピーすれば他プロジェクトでも動きます。
 *
 * 使い方:
 *   const seg = await createSkySegmenter();
 *   const mask = await seg.segment(videoElement);   // { width, height, data }
 *   // mask.data[y * mask.width + x] = 0.0(空でない) 〜 1.0(空)
 */

export const SKY_SEGMENTER_DEFAULTS = {
  // ---- ここを差し替えればモデルを変更できます ----
  modelUrl: './models/pp_mobileseg_tiny_ade20k_512.onnx',
  inputSize: 512, // モデルの入力解像度（この ONNX は固定 512x512）
  skyClassIndex: 2, // ADE20K の 150 クラス中 2 番が "sky"
  numClasses: 150,
  // ---------------------------------------------
  ortWasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/',
};

const MEAN = [0.485, 0.456, 0.406]; // ImageNet 正規化（学習時と同じ前処理）
const STD = [0.229, 0.224, 0.225];

export async function createSkySegmenter(options = {}) {
  const cfg = { ...SKY_SEGMENTER_DEFAULTS, ...options };

  // GitHub Pages は COOP/COEP ヘッダを付けられない = SharedArrayBuffer が使えないため、
  // wasm はシングルスレッドに固定する（自動判定に任せると初期化に失敗することがある）。
  ort.env.wasm.wasmPaths = cfg.ortWasmPaths;
  ort.env.wasm.numThreads = 1;
  // 推論を Web Worker 側で実行する。これがないと推論中(数百ms)メインスレッドが
  // 止まり、カメラ映像の描画がその間フリーズする。
  ort.env.wasm.proxy = true;

  const session = await ort.InferenceSession.create(cfg.modelUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });

  const size = cfg.inputSize;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  /**
   * @param {CanvasImageSource} source video / canvas / img / ImageBitmap
   * @returns {Promise<{width:number, height:number, data:Float32Array}>} 空である確率(0..1)の低解像度マップ
   */
  async function segment(source) {
    // 1. 入力サイズに縮小して画素を取り出す
    ctx.drawImage(source, 0, 0, size, size);
    const { data: rgba } = ctx.getImageData(0, 0, size, size);

    // 2. NCHW の Float32 に詰め替えつつ正規化
    // ※ Worker 実行時に ArrayBuffer が転送されるため、毎回確保し直す
    const input = new Float32Array(3 * size * size);
    const plane = size * size;
    for (let i = 0, p = 0; i < plane; i++, p += 4) {
      input[i] = (rgba[p] / 255 - MEAN[0]) / STD[0];
      input[i + plane] = (rgba[p + 1] / 255 - MEAN[1]) / STD[1];
      input[i + plane * 2] = (rgba[p + 2] / 255 - MEAN[2]) / STD[2];
    }

    // 3. 推論。出力は低解像度のロジット [1, 150, h, w]
    const outputs = await session.run({
      [inputName]: new ort.Tensor('float32', input, [1, 3, size, size]),
    });
    const logits = outputs[outputName];
    const [, numClasses, h, w] = logits.dims;
    const values = logits.data;

    // 4. 「空」と「空以外の最大」の 2 値ソフトマックス = sigmoid(sky - maxOther)
    //    150 クラス全部の softmax より安く、境界がなめらかな確率になる。
    const area = h * w;
    const prob = new Float32Array(area);
    for (let i = 0; i < area; i++) {
      const sky = values[cfg.skyClassIndex * area + i];
      let other = -Infinity;
      for (let c = 0; c < numClasses; c++) {
        if (c === cfg.skyClassIndex) continue;
        const v = values[c * area + i];
        if (v > other) other = v;
      }
      prob[i] = 1 / (1 + Math.exp(other - sky));
    }
    return { width: w, height: h, data: prob };
  }

  return {
    segment,
    inputSize: size,
    dispose: () => session.release?.(),
  };
}
