# tools/

`convert.py` は同梱モデル `models/pp_mobileseg_tiny_ade20k_512.onnx` を再生成するための
スクリプトです。**サンプルを動かすだけなら実行不要**です。

```bash
uv venv --python 3.12 tools/.venv
uv pip install --python tools/.venv/bin/python \
    paddlepaddle paddle2onnx onnx onnxruntime onnxslim pillow numpy \
    pyyaml opencv-python-headless scipy prettytable filelock six requests \
    tqdm scikit-learn scikit-image setuptools
tools/.venv/bin/python tools/convert.py
```

- paddle2onnx の macOS wheel は cp38〜cp312 のみ（**Python 3.13/3.14 では入りません**）
- 作業ファイルは `tools/_work/` に置かれます（`.gitignore` 済み）
- 検証結果のオーバーレイ画像が `tools/_work/validation.png` に出力されます
