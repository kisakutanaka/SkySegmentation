# 同梱モデルについて

`pp_mobileseg_base_ade20k_512.onnx` (約 22.6 MB)

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

## ライセンスに関する注意（重要）

- **モデルの重みは Apache-2.0** で配布されています（配布元: PaddlePaddle）。商用利用を許諾しています。
- 一方で**学習データである ADE20K の画像そのもの**は、MIT CSAIL により
  「非商用の研究・教育目的」に限定されています（アノテーションとソフトウェアは BSD-3）。
  学習済み重みにこの制限が及ぶかは見解が分かれます。
- 厳密なクリーンさが要求される用途では、自前データ or ライセンス許諾済みデータで
  学習し直したモデルに差し替えてください。差し替えは `sky-segmenter.js` の
  `SKY_SEGMENTER_DEFAULTS`（`modelUrl` / `inputSize` / `skyClassIndex`）を書き換えるだけです。

## 採用しなかったモデル（実測値つき）

| モデル | サイズ | 推論(1スレッド native) | 却下理由 |
|---|---|---|---|
| PP-MobileSeg-**Tiny** | 6.0 MB | 31 ms | マスクが 32×32 と粗く、稜線がにじむ |
| **SkySeg** (U-2-Net, [HF](https://huggingface.co/JianyuanWang/skyseg)) | 176 MB | **1250 ms** | 品質は最良だがリアルタイム不可。ONNX Runtime Web の WebGPU は `MaxPool(ceil_mode)` 未対応で動かず、wasm では 1 枚 3〜5 秒。ライセンス表記もなし |
| SegFormer-B0 (ADE20K) | 15 MB | — | 重みが NVIDIA Source Code License（**非商用限定**）。`nvidia/mit-b*` 派生も同様 |
| PIDNet / DDRNet / BiSeNet | — | — | MIT だが Cityscapes 学習（データセットが非商用限定） |
| HF の ONNX セグメンテーション上位 60 件 | — | — | Apache/MIT のものは人物切り抜き・背景除去・salient object 系のみで `sky` クラスがない |
