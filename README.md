# Heretic Chat - Uncensored Mistral 7B

Mistral 7B with censorship removed via [Heretic](https://github.com/p-e-w/heretic) directional ablation.

## What's inside

- `adapter/` - LoRA adapter weights (3x ablation, 4/100 refusals baseline)
- `scripts/heretic_api.py` - OpenAI-compatible API server
- `chat-app/` - Web chat interface with conversation memory
- `scripts/heretic_save*.py` - Scripts to reproduce the ablation

## Quick Start

1. **First time only**: Run `setup.bat` to install dependencies
2. **Start**: Run `start.bat` - launches API + chat UI
3. **Open**: http://localhost:3333
4. **Stop**: Run `stop.bat`

## Requirements

- Windows 10/11
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 5060)
- Python 3.11+
- Node.js 18+
- PyTorch nightly with CUDA 12.8 (cu128)

## Model Details

- **Base**: mistralai/Mistral-7B-Instruct-v0.3
- **Quantization**: BNB 4-bit
- **Ablation**: 3x weight multiplier on best Optuna trial (#61)
- **Result**: 0.6% refusal rate (down from 79%)
- **KL Divergence**: 0.63 (model capabilities preserved)

## Architecture

```
[Browser :3333] -> [Node.js Chat Server] -> [Python API :8000] -> [Mistral 7B + LoRA]
```

## Reproducing

To re-run the Heretic optimization from scratch:

```bash
set PYTHONPATH=D:\heretic\python-libs
set HF_HOME=D:\heretic\hf-cache
python -c "from heretic.main import main; import sys; sys.argv=['heretic','--model','mistralai/Mistral-7B-Instruct-v0.3','--quantization','BNB_4BIT','--max-batch-size','64']; main()"
```

This runs 200 Optuna trials (~4 hours on RTX 5060).
