# tools/

このディレクトリのスクリプトは**サンプルの実行には不要**です。同梱モデルをどう作ったかを
再現できるように置いてあります。

```bash
uv venv --python 3.12 tools/.venv
uv pip install --python tools/.venv/bin/python \
    onnxruntime numpy pillow onnx onnxslim torch
```

## A. 教師モデルの用意（`convert.py`）

PaddleSeg の PP-MobileSeg-Base (ADE20K / Apache-2.0) を ONNX に変換します。
paddle 関連の依存が追加で必要です（`convert.py` の docstring 参照）。

```bash
tools/.venv/bin/python tools/convert.py      # → models/pp_mobileseg_base_ade20k_512.onnx
```

## B. 空専用の小型モデルを蒸留する

150 クラスのモデルは「空か否か」の 1 ビットを出すには過剰なので、
教師の出力を使って二値専用の小型 CNN を学習します。

### B-1. 学習用画像を集める

Open Images V7 から「空」と「市街地（建物・高層ビル・通り・塔・信号など）」の
両方のラベルが付いた画像を選びます。**画像は CC BY 2.0 なので商用利用できます。**

```bash
# ラベル CSV(2.7GB) をストリーム処理して該当する画像 ID を抽出（約19万件）
curl -sL https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv \
  | python3 tools/filter_ids.py > dataset/train_ids.txt

# S3 から取得して 640px に縮小（S3 に存在しない ID が 2/3 ほどあるので多めに指定する）
tools/.venv/bin/python tools/fetch_images.py train dataset/train_ids.txt 12000 dataset/images/train
```

### B-2. 擬似ラベルを作る

教師は空専用の [SkySeg](https://huggingface.co/JianyuanWang/skyseg)（U-2-Net, 168MB, MIT）です。
リポジトリには含めていないので取得してください。

```bash
curl -L -o dataset/skyseg.onnx \
    https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx
```

PNG の R チャンネルに空確率、G チャンネルに教師の確信度を入れます。
SkySeg は 320×320 のソフトマットを返すので、確率そのものが境界の曖昧さを表しており、
確信度は一律 255（ソフトターゲットとして学習させる）です。

```bash
# 1枚 0.4 秒ほどかかるので、シャード番号とシャード数を渡して並列に回す
for k in 0 1 2 3; do
  tools/.venv/bin/python tools/pseudo_label_skyseg.py dataset/skyseg.onnx \
      dataset/images/train dataset/labels_skyseg/train $k 4 &
done; wait
```

旧版（ADE20K 由来の PP-MobileSeg-Base を教師にしたもの）は `pseudo_label.py` です。
こちらは 64×64 の出力をガイデッドフィルタで 256 に起こすため、境界が甘くなります。

```bash
tools/.venv/bin/python tools/pseudo_label.py \
    models/pp_mobileseg_base_ade20k_512.onnx dataset/images/train dataset/labels/train
```

### B-3. 学習する

`model.py` の `TinySkyNet` は depthwise separable conv だけの UNet で **49,233 パラメータ**。
入力 256×256、出力 128×128 のロジット 1 チャンネルです。

```bash
# train.py [エポック数] [ラベルのディレクトリ] [チェックポイントの保存先]
PYTHONPATH=tools tools/.venv/bin/python tools/train.py 40 \
    dataset/labels_skyseg dataset/tinyskynet_skyseg.pt
PYTHONPATH=tools tools/.venv/bin/python tools/export_onnx.py \
    dataset/tinyskynet_skyseg.pt models/tinyskynet_skyseg_256.onnx
```

毎エポック `<保存先>.last` に optimizer と scheduler ごと書き出すので、
途中で止めて同じコマンドを打てば続きから再開します。
`<保存先>` 本体には val IoU が最良のときだけ重みが入ります（`export_onnx.py` はこちらを読む）。

### B-4. 教師と比べる

```bash
tools/.venv/bin/python tools/compare.py \
    models/pp_mobileseg_base_ade20k_512.onnx models/tinyskynet_skyseg_256.onnx \
    dataset/images/val 8 /tmp/compare.png
```

## C. ブラウザでの動作確認（`cdp_shot.mjs`）

ヘッドレス Chrome でページを開き、スクリーンショットを撮ります。
`--use-file-for-fake-video-capture` に y4m を渡すと、カメラ映像を差し替えて
実機なしで合成結果を確認できます。

```bash
python3 -m http.server 8765 &
ffmpeg -loop 1 -i sky_photo.jpg -t 3 -r 15 -pix_fmt yuv420p -vf scale=640:480 fake_cam.y4m
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu \
  --remote-debugging-port=9333 --user-data-dir=/tmp/chrome-prof \
  --use-fake-ui-for-media-capture --use-fake-device-for-media-stream \
  --auto-accept-camera-and-microphone-capture --use-file-for-fake-video-capture=$PWD/fake_cam.y4m \
  about:blank &
node tools/cdp_shot.mjs http://localhost:8765/index.html /tmp/shot.png 20000
```

## 注意

- paddle2onnx の macOS wheel は cp38〜cp312 のみ（**Python 3.13/3.14 では入りません**）
- `dataset/` は `.gitignore` 済みです（数百 MB になります）
