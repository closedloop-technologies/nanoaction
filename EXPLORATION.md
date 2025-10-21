# Exploraiton of nanochat

## Tokenizer

```bash
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
python -m nanochat.dataset -n 8
```

```bash
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval
```

## Video models

OpenSora
Wan2.1

- <https://github.com/Wan-Video/Wan2.1/tree/main>

VideoGPT

- <https://arxiv.org/pdf/2104.10157>
- <https://wilsonyan.com/videogpt/index.html>
- <https://github.com/wilson1yan/videogpt>

Since we want to keep this small, we should start with YOLO or similar lightweight models and bake in historical frames in a way similar to OpenSora.
