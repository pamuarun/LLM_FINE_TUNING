# SQL Fine-Tuning with QLoRA — Spider Dataset

Fine-tuning Qwen 1.5B for Text-to-SQL generation using QLoRA on the Spider dataset.

## Results

| Metric | Baseline | Fine-Tuned | Improvement |
|---|---|---|---|
| Exact Match | 0.0% | 6.0% | +6.0% |
| Token Match | 34.39% | 54.75% | +20.36% |
| MMLU (forgetting check) | - | 16.0% | - |

## Model

Published on Hugging Face: [faltooz123/qwen1.5-sql-qlora-spider](https://huggingface.co/faltooz123/qwen1.5-sql-qlora-spider)

## Experiment Tracking

Training logs on Weights & Biases: [W&B Project](https://wandb.ai)

## Project Structure

```
D:/Project/
├── 01_dataset_preparation.ipynb   # Phase 1 - Dataset prep
├── 02_baseline_evaluation.ipynb   # Phase 2 - Baseline scoring
├── 03_qlora_finetuning.ipynb      # Phase 3 - QLoRA training
├── 04_evaluation.ipynb            # Phase 4 - Post-training eval
├── 05_publishing.ipynb            # Phase 5 - HF Hub publishing
├── data/
│   ├── spider_formatted/          # Processed dataset
│   ├── baseline_results.json      # Baseline scores
│   ├── finetuned_results.json     # Fine-tuned scores
│   └── comparison_results.json    # Before vs after
├── outputs/
│   └── qwen-sql-qlora/
│       └── final_adapter/         # LoRA adapter weights
└── requirements.txt
```

## Setup & Reproduction

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/qwen1.5-sql-qlora-spider
cd qwen1.5-sql-qlora-spider

# 2. Create virtual environment
python -m venv sql_finetune_env
sql_finetune_env\Scriptsctivate  # Windows

# 3. Install PyTorch (CUDA 12.8 for RTX 5060)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 4. Install dependencies
pip install -r requirements.txt

# 5. Login
wandb login
huggingface-cli login

# 6. Run notebooks in order (01 → 02 → 03 → 04 → 05)
jupyter notebook
```

## Hardware

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)
- Training time: ~15 minutes
- Method: QLoRA (4-bit quantization + LoRA rank 16)

## Key Findings

- Base model outputs explanations instead of clean SQL (0% exact match)
- After fine-tuning: token match improved by +20.36%
- Only 500 samples and 2 epochs — longer training would improve exact match further
- No catastrophic forgetting detected (MMLU: 16.0%)
