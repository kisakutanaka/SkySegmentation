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
tools/              モデルを再生成するためのスクリプト群（通常は実行不要）
```

## 同梱モデル

ページが読み込むのは **`models/tinyskynet_sky_256.onnx` の 199KB だけ**です。

| 項目 | 値 |
|---|---|
| サイズ | **199 KB** |
| パラメータ数 | 49,233（depthwise separable conv のみの UNet）|
| 入力 / 出力 | `[1,3,256,256]` → `[1,1,128,128]` のロジット |
| 推論（Chrome / wasm 1スレッド） | **約 12 ms** |

ADE20K 150 クラスのモデル（PP-MobileSeg-Base, 22.6MB）を教師にして、
「空か否か」の 1 ビットだけを出すように蒸留したものです。
`models/` にはその教師も置いてありますが、**擬似ラベルを作り直すとき以外は使いません**。
作り方は [tools/README.md](tools/README.md)、判断材料は [models/README.md](models/README.md) を参照してください。

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

## デバッグ表示

画面上部の「空領域を塗りつぶして表示（デバッグ）」にチェックを入れると、
合成をやめてマスクそのもの（空と判定された領域）をシアンで塗って表示します。
モデルの誤判定や、境界がどこまで追従しているかを確認するのに使えます。

## 仕組み

### 1. 空マスクの推論（`sky-segmenter.js`）

空/非空の二値モデルを onnxruntime-web で実行して確率マップを得ます。

- 前処理: 512×512 に縮小 → RGB を 0-1 → ImageNet 正規化 → NCHW
- 後処理: 出力ロジット `[1,1,128,128]` を sigmoid して空である確率（0〜1）にする
  - ADE20K 系の 150 クラスモデルに差し替えた場合は、
    `p = sigmoid(sky - max(その他のクラス) - skyMargin)` の経路が自動で使われます
    （出力チャンネル数で判定）。`skyMargin` は「空が他クラスに差をつけて勝ったときだけ
    空とみなす」ためのバイアスで、霞んだ遠景の地面が僅差で空と判定されるのを防ぎます
- **エッジ吸着**: 64×64 のままでは稜線がにじむため、映像の輝度をガイドにした
  [ガイデッドフィルタ](https://kaiminghe.github.io/eccv10/) を 256×256 で掛けて
  マスクを被写体の輪郭に吸着させます（+約 10 ms、モデルを重くせずに境界だけ改善できる）

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

マスクはガイデッドフィルタ後の 256×256 で、`drawImage` の拡大が
そのまま境界のフェザリングとして働きます。

## 他のプロジェクトで使うには

`sky-segmenter.js` と `models/tinyskynet_sky_256.onnx`（199KB）をコピーし、
onnxruntime-web を読み込むだけです。

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

| 環境 | マスク 1 枚（推論＋エッジ吸着） |
|---|---|
| Apple Silicon Mac / Chrome / wasm 1スレッド | 約 12 ms |
| 同 / Python onnxruntime 1スレッド（推論のみ） | 約 4.8 ms |

スマートフォンでは数倍かかりますが、上記のループ分離により**表示は 60fps のまま**です
（マスクの更新だけが数 fps になり、カメラを速く振ったときに少し遅れて追従します）。

精度を優先したい場合は、教師の PP-MobileSeg-Base に戻せます（`sky-segmenter.js` の
`modelUrl` を `./models/pp_mobileseg_base_ade20k_512.onnx`、`inputSize` を `512` にするだけ）。
22.6MB になり推論も 10 倍かかりますが、明るい壁面の誤判定は減ります。

### さらに速くしたい場合

- `sky-segmenter.js` の `inputSize` を下げた ONNX に差し替える（`tools/convert.py` の `INPUT_SIZE`）
- GitHub Pages は COOP/COEP ヘッダを付けられないため wasm はシングルスレッド固定です。
  [`coi-serviceworker`](https://github.com/gzuidhof/coi-serviceworker) を置いてヘッダを注入すれば
  `ort.env.wasm.numThreads` を増やせます
- WebGPU 対応端末では `executionProviders: ['webgpu', 'wasm']` を試す

## ライセンス

- **コード**: MIT（[LICENSE](LICENSE)）
- **同梱モデル**:
  - TinySkyNet（ページが読み込むもの）: MIT（このリポジトリで学習。学習画像は Open Images の CC BY 2.0 写真）
  - PP-MobileSeg-Base（教師。擬似ラベル生成用）: Apache-2.0（PaddleSeg）
  — 学習データに関する注意を含め、[models/README.md](models/README.md) を必ずお読みください
