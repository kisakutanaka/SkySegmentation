# Sky Segmentation Composite — ブラウザだけで空を分離して合成するサンプル

スマホ／PC のブラウザでカメラ映像から**空**を分離し、
**後景（空）→ スマイリー → 前景（空以外）** の順に重ねてリアルタイム合成します。
スマイリーは空に浮かび、建物や山の背後に回り込みます。

- ビルド不要・npm 不要。**静的ファイルを置くだけ**（GitHub Pages でそのまま動きます）
- 推論は端末内で完結。画像はサーバーに送られません
- 推論部分は [`sky-segmenter.js`](sky-segmenter.js) の 1 ファイルに閉じているので、他プロジェクトにコピーして使えます

## ファイル構成

```
index.html          UI（canvas と開始ボタンだけ）
app.js              カメラ取得・描画ループ・合成
sky-segmenter.js    ★ 空マスク推論だけの独立モジュール（依存は onnxruntime-web のみ）
models/             同梱 ONNX モデルとそのライセンス
tools/convert.py    モデルを再生成するための変換スクリプト（通常は実行不要）
```

## 使い方（ローカル）

```bash
python3 -m http.server 8000
# http://localhost:8000 を開く（localhost は secure context 扱いなのでカメラが使えます）
```

スマホの実機で試す場合は **HTTPS が必須**です。LAN 経由の平文 HTTP では
`getUserMedia` がブロックされるため、GitHub Pages に公開してから開いてください。

## GitHub Pages に公開する

`.github/workflows/pages.yml` を同梱してあります。

1. このディレクトリを GitHub リポジトリとして push する
   ```bash
   git init -b main
   git add .
   git commit -m "Add sky segmentation composite sample"
   gh repo create <リポジトリ名> --public --source=. --push
   ```
2. リポジトリの **Settings › Pages › Build and deployment › Source** を
   **GitHub Actions** に設定する
3. push するたびに自動デプロイされ、`https://<user>.github.io/<repo>/` で公開されます

（Actions を使わず **Deploy from a branch** でも動きます。`.nojekyll` を同梱済みなので追加設定は不要です。）

## 仕組み

### 1. 空マスクの推論（`sky-segmenter.js`）

ADE20K 150 クラスのセマンティックセグメンテーションモデルを onnxruntime-web で実行し、
**クラス 2 = `sky`** の確率だけを取り出します。

- 前処理: 512×512 に縮小 → RGB を 0-1 → ImageNet 正規化 → NCHW
- 後処理: 出力ロジット `[1,150,32,32]` から
  `p = sigmoid(sky - max(その他のクラス))` で空である確率（0〜1）を作る
  - 150 クラスの softmax を全部計算するより安く、境界がなめらかになります

### 2. 合成（`app.js`）

**推論ループと描画ループを分離**しているのがポイントです。

- 推論ループ: 端末性能なりの速度（数 fps）で回り、最新のマスクだけを更新する
- 描画ループ: `requestAnimationFrame` で 30〜60 fps。**常に最新マスクを使い回す**ので
  推論が遅い端末でも映像はカクつきません
- 推論は `ort.env.wasm.proxy = true` により Web Worker 側で走るので、メインスレッドは止まりません

1 フレームの合成は Canvas 2D だけで完結します。

```
① view に カメラ映像 を描く            ← 後景（空を含む）
② その上に 😀 を描く
③ 作業用 canvas に カメラ映像 を描き、
   globalCompositeOperation='destination-in' でマスクを重ねて 前景だけ を残す
④ ③ を view の最前面に描く            ← スマイリーが建物や山の背後に回り込む
```

マスクは 32×32 と粗いですが、`drawImage` の拡大（バイリニア補間）が
そのまま境界のフェザリングとして働きます。

## 他のプロジェクトで使うには

`sky-segmenter.js` と `models/` をコピーし、onnxruntime-web を読み込むだけです。

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.min.js"></script>
<script type="module">
  import { createSkySegmenter } from './sky-segmenter.js';

  const segmenter = await createSkySegmenter();          // モデル読み込み
  const mask = await segmenter.segment(videoOrImage);    // { width, height, data }
  // mask.data[y * mask.width + x] = 0.0(空でない) 〜 1.0(空)
</script>
```

モデルを差し替えるときは `SKY_SEGMENTER_DEFAULTS` の
`modelUrl` / `inputSize` / `skyClassIndex` を変更してください。

## 実測値（参考）

| 環境 | 推論 1 回 |
|---|---|
| Apple Silicon Mac / Chrome / wasm 1スレッド | 約 50〜60 ms |
| 同 / Python onnxruntime 1スレッド | 約 31 ms |

スマートフォンでは数倍かかりますが、上記のループ分離により表示は滑らかなままです。

### さらに速くしたい場合

- `sky-segmenter.js` の `inputSize` を下げた ONNX に差し替える（`tools/convert.py` の `INPUT_SIZE`）
- GitHub Pages は COOP/COEP ヘッダを付けられないため wasm はシングルスレッド固定です。
  [`coi-serviceworker`](https://github.com/gzuidhof/coi-serviceworker) を置いてヘッダを注入すれば
  `ort.env.wasm.numThreads` を増やせます
- WebGPU 対応端末では `executionProviders: ['webgpu', 'wasm']` を試す

## ライセンス

- **コード**: MIT（[LICENSE](LICENSE)）
- **同梱モデル**: Apache-2.0（PaddleSeg / PP-MobileSeg-Tiny）
  — 学習データ ADE20K に関する注意を含め、[models/README.md](models/README.md) を必ずお読みください
