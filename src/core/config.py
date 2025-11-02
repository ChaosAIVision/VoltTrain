from pydantic import BaseModel

class ModelConfig(BaseModel):

    model_name_or_path:str 
    load_4bit:bool = False
    load_8bit:bool = False
    load_fp8:bool = False
    load_fp4:bool = False

class LoraConfig(BaseModel):
    lora_r:int = 16
    lora_alpha:int = 32
    lora_dropout:float = 0.05
    lora_target_modules:list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_bias:str = "none"
    lora_task_type:str = "CAUSAL_LM"
    full_finetuning:bool = False

class DatasetConfig(BaseModel):
    dataset_name_or_path:str = ""
    dataset_type:str = ""
    split:str = "train"

class TrainingConfig(BaseModel):
    train_batch_size:int = 1
    eval_batch_size:int = 1
    gradient_accumulation_steps:int = 1
    gradient_checkpointing:bool = True
    learning_rate:float = 2e-4
    weight_decay:float = 0.0
    max_grad_norm:float = 1.0
    max_steps:int = -1
    warmup_steps:int = 0
    logging_steps:int = 10
    save_steps:int = 1000
    save_total_limit:int = 10
    eval_steps:int = 1000
    eval_accumulation_steps:int = 1
    eval_delay:int = 0
    fp16:bool = False
    bf16:bool = False
    tf32:bool = False
    local_rank:int = -1
    ddp_backend:str = "nccl"
    ddp_find_unused_parameters:bool = False
    ddp_timeout:int = 1800
    ddp_bucket_cap_mb:int = 25
    ddp_broadcast_buffers:bool = False
    gpu_devices:list[int] = [0]
    wandb_project:str = ""
    wandb_run_name:str = ""
    epochs:int = 3
    output_dir:str = "outputs"
    num_checkpoint:int = -1
    muon_lr:float = 0.02
    muon_momentum:float = 0.95
    
