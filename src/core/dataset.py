from datatasets import load_dataset
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
import torch

def load_dataset(dataset_name_or_path:str, split:str, range:Tuple[int,int]=None, is_json:bool=False):

    """
    Load dataset from dataset_name_or_path.

    Args:
        dataset_name_or_path (str): Dataset name or path.
        split (str): Split name.
        range (Tuple[int,int], optional): Range of dataset. Defaults to None.
        is_json (bool, optional): Whether dataset is json file. Defaults to False.

    Returns:
        Dataset: Dataset.
    """

    if is_json:
        dataset = load_dataset(dataset_name_or_path, split=split, format="json")
    else:
        dataset = load_dataset(dataset_name_or_path, split=split)
    if range is not None:
        dataset = dataset.select(range(range[0], range[1]))

    return dataset


def build_custom_chat_template(messages: List[Dict[str, Any]]) -> str:

    """
    Build custom chat template from messages because some model 
    require special chat template.

    Args:
        messages (List[Dict[str, Any]]): Messages.

    Returns:
        str: Custom chat template.
    """

    messages = example["messages"]
    text = ""

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()

        if role == "system":
            text += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>"
        elif role == "user":
            text += f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>"
        elif role == "assistant":
            text += f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>"

    text += "<|eot_id|>"
    return {"text": text}




def build_chat_template(messages: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> str:
    """ 
    Build chat template from messages with default tokenizer from huggingface.

    Args:
        messages (List[Dict[str, Any]]): Messages.
        tokenizer (AutoTokenizer): Tokenizer.

    Returns:
        str: Custom chat template.
    """
    text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )

    # Set default eos_token if not exist
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|end|>"
    # Add eos_token if not exist
    if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    return text


class SFTDataset(Dataset):

    def __init__(self, dataset: Dataset, tokenizer: AutoTokenizer, args):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.dataset)

    
    def preprocess_data(self, examples: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        """
        Preprocess data.
        """

        if self.args.custom_chat_template:
            examples["text"] = [build_custom_chat_template(messages, self.tokenizer) for messages in examples["messages"]]
        else:
            examples["text"] = [build_chat_template(messages, self.tokenizer) for messages in examples["messages"]]

        # Tokenize text
        tokenized_examples = self.tokenizer(
            examples["text"],
            padding="max_length",
            max_length=self.args.max_seq_length,
            truncation=True,
            return_tensors="pt"
        )

        return 
        {
            "input_ids": tokenized_examples["input_ids"],
            "attention_mask": tokenized_examples["attention_mask"],
            "labels": tokenized_examples["input_ids"].clone()
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get item.
        """
        return self.preprocess_data(self.dataset[index])

    
    
