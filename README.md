# mteb-nl-dev

**MTEB-NL** is a massive text embedding benchmark for Dutch.
It extends the [MTEB (Massive Text Embedding Benchmark)](https://github.com/embeddings-benchmark/mteb) with tasks and datasets tailored to Dutch (NL).  

⚠️ **Note**: This repository is temporary and will be deprecated once **mteb-nl-dev** is incorporated into the main [MTEB](https://github.com/embeddings-benchmark/mteb) project.

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/nikolay-banar/mteb-nl-dev.git
cd mteb-nl-dev
pip install -r requirements.txt
```
## 🚀 Usage
Evaluate a multilingual model (already included in MTEB)
```bash
CUDA_VISIBLE_DEVICES=0 python run_eval.py \
  --task_name "MTEB(nl, v1)" \
  --batch_size 64 \
  --model_name intfloat/multilingual-e5-base

```
Evaluate Dutch supervised models (introduced in our paper)
```bash
CUDA_VISIBLE_DEVICES=0 python run_eval.py \
  --task_name "MTEB(nl, v1)" \
  --batch_size 64 \
  --model_name "clips/e5-base-trm-nl" \
  --model_revision "0645a0c0cbe062c6fc396e54ca6cc5689eb6e79f" \
  --model_type "e5"

```

Evaluate Dutch self-supervised models
```bash

CUDA_VISIBLE_DEVICES=0 python run_eval.py \
  --task_name "MTEB(nl, v1)" \
  --batch_size 64 \
  --model_name "DTAI-KULeuven/robbert-2022-dutch-base" \
  --model_revision "5405706bb8e38674356e19a345029492ab6e0aea" \
  --model_type "mean" 

```
## 📖 Citation
If you use **MTEB-NL** in your research, please cite:
```bibtex
@misc{banar2025mtebnle5nlembeddingbenchmark,
      title={MTEB-NL and E5-NL: Embedding Benchmark and Models for Dutch}, 
      author={Nikolay Banar and Ehsan Lotfi and Jens Van Nooten and Cristina Arhiliuc and Marija Kliocaite and Walter Daelemans},
      year={2025},
      eprint={2509.12340},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.12340}, 
}
```