# 同梱モデルについて

`pp_mobileseg_tiny_ade20k_512.onnx` (約 6.0 MB)

| 項目 | 内容 |
|---|---|
| 元モデル | [PP-MobileSeg-Tiny](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/configs/pp_mobileseg) (PaddleSeg) |
| ライセンス | **Apache-2.0**（`LICENSE-PaddleSeg.txt` 参照） |
| 学習データ | ADE20K（150クラス、`sky` はインデックス 2） |
| パラメータ数 | 1.61 M / ADE20K mIoU 36.39% |
| 入力 | `x` : `[1, 3, 512, 512]` float32、RGB を 0-1 にして ImageNet 正規化 |
| 出力 | `[1, 150, 32, 32]` の低解像度ロジット |
| 変換 | `tools/convert.py`（Paddle → ONNX → onnxslim。最終アップサンプルは除去済み） |

## ライセンスに関する注意（重要）

- **モデルの重みは Apache-2.0** で配布されています（配布元: PaddlePaddle）。商用利用を許諾しています。
- 一方で**学習データである ADE20K の画像そのもの**は、MIT CSAIL により
  「非商用の研究・教育目的」に限定されています（アノテーションとソフトウェアは BSD-3）。
  学習済み重みにこの制限が及ぶかは見解が分かれます。
- 厳密なクリーンさが要求される用途では、自前データ or ライセンス許諾済みデータで
  学習し直したモデルに差し替えてください。差し替えは `sky-segmenter.js` の
  `SKY_SEGMENTER_DEFAULTS`（`modelUrl` / `inputSize` / `skyClassIndex`）を書き換えるだけです。

## 参考: 採用しなかったモデル

- **SegFormer-B0 (ADE20K)** — 精度・軽さともに良いが、重みが NVIDIA Source Code License
  （非商用限定）。`nvidia/mit-b*` を親に持つ派生モデルも同様。
- **SkyAR / U-2-Net 系の空分離モデル** — 176MB あり GitHub の 100MB/ファイル制限を超える。
- **PIDNet / DDRNet / BiSeNet** — MIT だが Cityscapes 学習（データセットが非商用限定）。
- **Hugging Face の ONNX セグメンテーションモデル** — Apache/MIT のものは人物切り抜き・
  背景除去・salient object 系ばかりで、`sky` クラスを持つものが見当たらない。
