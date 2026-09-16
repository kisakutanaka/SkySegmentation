# 同梱モデルについて

ページが実際に読み込むのは **`tinyskynet_skyseg_256.onnx` (199KB)** だけです。
ほかの 2 つは学習・比較用で、ブラウザからは読み込まれません。

| ファイル | 役割 |
|---|---|
| `tinyskynet_skyseg_256.onnx` | **ページが読み込むモデル**。教師は SkySeg |
| `tinyskynet_sky_256.onnx` | 旧版。教師は PP-MobileSeg-Base（ADE20K）。比較用に残してある |
| `pp_mobileseg_base_ade20k_512.onnx` | 旧版の教師 |

教師の **SkySeg (168MB)** はサイズの都合でリポジトリに含めていません。
`tools/README.md` の手順で取得してください。

## `tinyskynet_skyseg_256.onnx` (199 KB) — 現行

| 項目 | 内容 |
|---|---|
| 出所 | **このリポジトリで学習**（`tools/train.py`）。SkySeg を教師にした蒸留 |
| 教師 | [SkySeg](https://huggingface.co/JianyuanWang/skyseg)（U-2-Net, 168MB, **MIT**。出所は [xiongzhu666/Sky-Segmentation-and-Post-processing](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing)）|
| ライセンス | **MIT**（重み・コードとも。ただし下記の学習データの注記を参照） |
| 学習データ | Open Images V7 のうち「空」と「市街地」の両方のラベルが付いた写真 3,901 枚。**画像は CC BY 2.0**（商用可）|
| パラメータ数 | 49,233（旧版と同じ構造）|
| 入力 / 出力 | `[1, 3, 256, 256]` → `[1, 1, 128, 128]` のロジット |

旧版との違いは**教師だけ**です。PP-MobileSeg-Base は 64×64 の出力をガイデッドフィルタで
起こしたものを擬似ラベルにしていたため、生徒が学べる輪郭の細かさがそこで頭打ちでした。
SkySeg は空専用の二値モデルで 320×320 のソフトマットを返すので、ラベルの境界が正確です。

検証 335 枚を SkySeg の出力と突き合わせた一致度:

| モデル | 全体 IoU | 境界帯 ±8px の一致率 |
|---|---|---|
| 旧 `tinyskynet_sky_256` | 0.8479 | 0.8912 |
| 新 `tinyskynet_skyseg_256` | **0.8888** | **0.9223** |

※ 新モデルはこの SkySeg を教師にして学習しているので、**この指標は新モデルに有利**です。
「SkySeg の出力にどれだけ近いか」以上の意味はありません。

**既知の弱点**: 明るく平坦な壁面（白い建物など）を空と誤判定することがあります。
容量不足（49K パラメータ）が理由なので、チャンネル数を増やせば改善する見込みです。

## `tinyskynet_sky_256.onnx` (199 KB) — 旧版

| 項目 | 内容 |
|---|---|
| 出所 | 同じ構造を PP-MobileSeg-Base（下記）の擬似ラベルで学習したもの |
| ライセンス | **MIT**（重み。ただし教師が ADE20K 学習である点は下記の注記を参照） |
| 学習データ | 上と同じ Open Images の写真 3,839 枚 |

---

## `pp_mobileseg_base_ade20k_512.onnx` (約 22.6 MB)

| 項目 | 内容 |
|---|---|
| 元モデル | [PP-MobileSeg-Base](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/configs/pp_mobileseg) (PaddleSeg) |
| ライセンス | **Apache-2.0**（`LICENSE-PaddleSeg.txt` 参照） |
| 学習データ | ADE20K（150クラス、`sky` はインデックス 2） |
| パラメータ数 | 5.62 M / ADE20K mIoU 41.57% |
| 入力 | `x` : `[1, 3, 512, 512]` float32、RGB を 0-1 にして ImageNet 正規化 |
| 出力 | `[1, 150, 64, 64]` の低解像度ロジット |
| 変換 | `tools/convert.py`（Paddle → ONNX → onnxslim。最終アップサンプルは除去済み） |

出力が 64×64 と粗いのは、境界の精緻化を `sky-segmenter.js` 側のガイデッドフィルタで
行っているためです（モデルを重くせずにエッジを立てられる）。

---

## ライセンスに関する注意（重要）

### PP-MobileSeg-Base

- **モデルの重みは Apache-2.0** で配布されています（配布元: PaddlePaddle）。商用利用を許諾しています。
- 一方で**学習データである ADE20K の画像そのもの**は、MIT CSAIL により
  「非商用の研究・教育目的」に限定されています（アノテーションとソフトウェアは BSD-3）。
  学習済み重みにこの制限が及ぶかは見解が分かれます。
- 厳密なクリーンさが要求される用途では、自前データ or ライセンス許諾済みデータで
  学習し直したモデルに差し替えてください。差し替えは `sky-segmenter.js` の
  `SKY_SEGMENTER_DEFAULTS`（`modelUrl` / `inputSize` / `skyClassIndex`）を書き換えるだけです。

### SkySeg（現行モデルの教師）

- 重みは **MIT**。HuggingFace の配布元・元リポジトリともに MIT を明記しています。
- ただし**学習データは非公開**です（作者は「高精度版は自社プロダクトで使っている」と述べています）。
  ADE20K のように「非商用限定」と明記されたデータではありませんが、**素性を検証できません**。
  既知の制限が無いかわりに未知が残る、という交換だと理解してください。

### TinySkyNet

- 重みは MIT ですが、**教師の出力を使って学習している**ため、教師の性質を引き継いでいると
  考えるのが安全です。現行モデルは SkySeg、旧版は PP-MobileSeg-Base（ADE20K）が教師です。
- 学習に使った Open Images の写真自体は CC BY 2.0 で商用利用できます。

## 採用しなかったモデル（実測値つき）

| モデル | サイズ | 推論(1スレッド native) | 却下理由 |
|---|---|---|---|
| PP-MobileSeg-**Tiny** | 6.0 MB | 31 ms | マスクが 32×32 と粗く、稜線がにじむ |
| **SkySeg** (U-2-Net, [HF](https://huggingface.co/JianyuanWang/skyseg)) | 168 MB | 400〜1250 ms | 品質は最良だが**ブラウザでは**リアルタイム不可（WebGPU は `MaxPool(ceil_mode)` 未対応、wasm では 1 枚 3〜5 秒）。**オフラインの教師としては採用**している |
| SegFormer-B0 (ADE20K) | 15 MB | — | 重みが NVIDIA Source Code License（**非商用限定**）。`nvidia/mit-b*` 派生も同様 |
| PIDNet / DDRNet / BiSeNet | — | — | MIT だが Cityscapes 学習（データセットが非商用限定） |
| HF の ONNX セグメンテーション上位 60 件 | — | — | Apache/MIT のものは人物切り抜き・背景除去・salient object 系のみで `sky` クラスがない |
