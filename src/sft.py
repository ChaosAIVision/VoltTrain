from src.core.dataset import SFTDataset
from src.core.model_ptln import LiSFT
from src.utils.loader.load_yaml import load_yaml_config
from src.utils.model_utils.checkpointing import EpochEndSaver
from src.utils.data_utils.process_data import collate_sft
from src.core.config import ModelConfig, LoraConfig, DatasetConfig, TrainingConfig
from src.utils.wandb import create_wandb_run

import torch
import wandb
import os 

from peft import LoraConfig as PeftLoraConfig, get_peft_model
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoModelForCausalLM, 
    AutoTokenizer)
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl


torch.cuda.empty_cache()
try:
    torch.cuda.set_per_process_memory_fraction(0.95, device=0)
except Exception:
    pass

def main():

    #################
    #   Load config  #
    #################
    config = load_yaml_config("config/sft.yaml")

    # Load model config
    model_config = ModelConfig(**config["model"])
    # Load lora config
    lora_config = LoraConfig(**config["lora"])
    # Load dataset config 
    dataset_config = DatasetConfig(**config["dataset"])
    # Load training config
    training_config = TrainingConfig(**config["training"])
    
    # Calculate number of devices
    num_devices = len(training_config.gpu_devices)

    # if use quantization method
    load_kwargs = {}
    if model_config.load_4bit or model_config.load_8bit or model_config.load_fp8 or model_config.load_fp4:
        load_kwargs = {}
        if model_config.load_4bit:
            load_kwargs.update(dict(load_in_4bit=True))
        if model_config.load_8bit:
            load_kwargs.update(dict(load_in_8bit=True))
        
    model  = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        device_map=None,
        trust_remote_code=True,
        **load_kwargs
    )

    # use gradient checkpointing 

    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            try:
                model.config.use_cache = False
            except Exception:
                pass

    target_modules = lora_config.lora_target_modules
    lconf = PeftLoraConfig(
        r=lora_config.lora_r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=target_modules,
        bias=lora_config.lora_bias,
        task_type=lora_config.lora_task_type,
    )

    model = get_peft_model(model, lconf)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=True,
    )

    lit_model = LiSFT(
        model = model,
        lr=training_config.learning_rate,
        grad_clip=training_config.max_grad_norm,
        use_muon=True,
        muon_lr=training_config.muon_lr,
        muon_momentum=training_config.muon_momentum,
        weight_decay=training_config.weight_decay
    )

    # Load dataset from HuggingFace
    from src.core.dataset import load_dataset
    raw_dataset = load_dataset(
        dataset_config.dataset_name_or_path,
        dataset_config.split
    )
    
    # Split dataset
    train_test_split = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    valid_dataset = train_test_split['test']
    
    # Create simple args object for SFTDataset
    class DatasetArgs:
        custom_chat_template = False
        max_seq_length = 512
    
    dataset_args = DatasetArgs()
    
    # Create dataset wrappers
    train_dataset_custom = SFTDataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        args=dataset_args,
    )
    valid_dataset_custom = SFTDataset(
        dataset=valid_dataset,
        tokenizer=tokenizer,
        args=dataset_args,
    )

    from src.utils.data_utils.process_data import get_data_collator
    data_collator = get_data_collator(tokenizer, mlm=False)
    
    train_dataloader = DataLoader(
        train_dataset_custom,
        batch_size=training_config.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    valid_dataloader = DataLoader(
        valid_dataset_custom,
        batch_size=training_config.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    epoch_saver_cb = EpochEndSaver(
        output_dir=training_config.output_dir,
        num_checkpoint=training_config.num_checkpoint,
    )

    # init wandb 
    wandb_logger = None
    if training_config.wandb_project:
        create_wandb_run(
            project_name=training_config.wandb_project,
            run_name=training_config.wandb_run_name,
            config=config,
        )
        wandb_logger = WandbLogger(project=training_config.wandb_project, name=training_config.wandb_run_name)


    trainer = pl.Trainer(
        max_epochs=training_config.epochs,
        default_root_dir=getattr(training_config, 'output_dir', 'outputs'),
        log_every_n_steps=getattr(training_config, 'log_every_n_steps', training_config.logging_steps),
        accumulate_grad_batches=training_config.gradient_accumulation_steps,
        precision=getattr(training_config, 'precision', 'bf16-mixed') if training_config.bf16 else ('16-mixed' if training_config.fp16 else '32'),
        enable_checkpointing=False,
        callbacks=[epoch_saver_cb],
        devices=training_config.gpu_devices,
        accelerator='gpu',
        strategy="auto" if num_devices == 1 else "ddp",
        logger=wandb_logger)

    trainer.fit(lit_model, 
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader)
    

    # Save adapter and tokenizer
    output_dir = getattr(training_config, 'output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    lit_model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()












    


    
