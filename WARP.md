# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

VoltTrain is a research codebase for training Large Language Models (LLMs) using:
- **Custom Muon optimizer** with Newton-Schulz orthogonalization
- **FP4/8-bit precision** training support
- **PyTorch Lightning** training framework
- **LoRA/PEFT** for parameter-efficient fine-tuning

## Commands

### Running Training
```bash
# Run supervised fine-tuning with default config
python -m src.sft

# Training uses config/sft.yaml for all hyperparameters
```

### Configuration
All training parameters are configured via `config/sft.yaml`. Key sections:
- `model`: Model path and quantization settings
- `lora`: LoRA configuration (rank, alpha, target modules)
- `dataset`: Dataset path and type
- `training`: Training hyperparameters, devices, W&B settings

## Architecture

### Core Training Flow
1. **Entry Point**: `src/sft.py` - Main training script
   - Loads YAML config and initializes all components
   - Supports 4/8-bit quantization via transformers
   - Uses PyTorch Lightning Trainer for multi-GPU support

2. **PyTorch Lightning Module**: `src/core/model_ptln.py` (`LiSFT` class)
   - Wraps HuggingFace models for PyTorch Lightning
   - Handles multi-optimizer setup (Muon for matrices, AdamW for embeddings)
   - Implements differentiated learning rates:
     - Matrix params: configured LR with Muon optimizer
     - Embeddings: 2x base LR
     - Unembedding/LM head: 0.1x base LR
   - Custom LR scheduling with warmup + cosine decay

3. **Muon Optimizer**: `src/core/optimizer.py`
   - Custom optimizer using Newton-Schulz orthogonalization
   - `zeropower_via_newtonschulz5()`: Performs NS5 iterations for gradient orthogonalization
   - **Do NOT use for embedding or final FC layers** - only for matrix parameters
   - Compiled with `@torch.compile` for performance

### Data Pipeline
- **Dataset Loading**: `src/core/dataset.py` (`SFTDataset` class)
  - Supports both custom and standard HuggingFace chat templates
  - `build_chat_template()`: Uses tokenizer's default template
  - `build_custom_chat_template()`: Custom template for special models
  - Auto-adds EOS tokens if missing

- **Data Processing**: `src/utils/data_utils/process_data.py`
  - `collate_sft()` is deprecated - use `get_data_collator()` instead
  - Returns HuggingFace `DataCollatorForLanguageModeling`

### Configuration System
- **Config Classes**: `src/core/config.py`
  - Pydantic models for type-safe configuration
  - Classes: `ModelConfig`, `LoraConfig`, `DatasetConfig`, `TrainingConfig`
  - Loaded via `load_yaml_config()` from `src/utils/loader/load_yaml.py`

### Utilities
- **Checkpointing**: `src/utils/model_utils/checkpointing.py` (`EpochEndSaver`)
  - PyTorch Lightning callback for epoch-based checkpointing
  - Saves PEFT adapters + tokenizer
  - Configurable checkpoint retention (keeps last N)

- **W&B Integration**: `src/utils/wandb.py`
  - `create_wandb_run()` initializes experiment tracking

- **LoRA Presets**: `src/utils/model_utils/mamba_layer.py`
  - Predefined target module presets for different architectures
  - `PRESETS`: mlp_only, mamba_only, mlp_mamba, attn_mlp_ssm, llama
  - Use `get_targets(preset_name)` to retrieve module lists

## Key Implementation Details

### Multi-Optimizer Strategy
The codebase uses separate optimizers for different parameter groups:
- **Muon**: Applied to matrix parameters (if `use_muon=True`)
- **AdamW/standard optimizer**: Applied to embeddings and unembedding layers
- Each group has differentiated learning rates optimized for LLM training

### Quantization
- Supports loading models in 4-bit, 8-bit via HuggingFace transformers
- Configure via `model.train_4bit`, `model.train_8bit` in YAML
- FP4/FP8 flags exist but implementation incomplete

### Gradient Checkpointing
- Enabled via `training.gradient_checkpointing: true`
- Automatically disables `model.config.use_cache` for compatibility
- Essential for training large models with limited VRAM

### Multi-GPU Training
- Uses PyTorch Lightning's DDP strategy automatically
- Configure devices via `training.gpu_devices: [0, 1, ...]`
- DDP settings in config: backend, timeout, bucket size, etc.

## Important Notes

- **Muon optimizer should only be used for matrix parameters**, not embeddings or output layers
- The codebase has some incomplete imports (e.g., missing `torch` import in `sft.py` line 25)
- Dataset loading function `load_dataset()` shadows the HuggingFace function name
- Training script references undefined variables: `num_devices`, `args.output_dir`, `wandb_logger`
