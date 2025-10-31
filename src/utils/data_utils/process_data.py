from transformers import AutoTokenizer


def get_data_collator(tokenizer: AutoTokenizer, mlm: bool = False):
    """
    Get data collator for language modeling.
    
    Args:
        tokenizer: HuggingFace tokenizer
        mlm: Whether to use masked language modeling (False for causal LM)
    
    Returns:
        DataCollatorForLanguageModeling instance
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,  # False for causal language modeling (SFT)
        mlm_probability=0.15 if mlm else None,
        pad_to_multiple_of=None,
        return_tensors="pt"
    )


# Deprecated: Use get_data_collator() instead
def collate_sft(batch: List[Dict[str, Any]], tokenizer: AutoTokenizer):
    """
    DEPRECATED: Use get_data_collator() instead.
    
    HuggingFace-style collator for causal language modeling using tokenizer.
    Uses tokenizer.pad() for proper padding and tensor conversion.
    """
    # Use get_data_collator() instead
    data_collator = get_data_collator(tokenizer, mlm=False)
    return data_collator(batch)