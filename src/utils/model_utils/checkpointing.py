from pytorch_lightning.callbacks import Callback
import os
import lightning  as  pl

class EpochEndSaver(Callback):
    """Save adapter + tokenizer at the end of each epoch with epoch index in filename."""
    def __init__(self, output_dir, num_checkpoint=-1):
        self.output_dir = output_dir
        self.num_checkpoint = num_checkpoint
        self.saved_epochs = []
    
    def on_train_epoch_end(self, trainer, pl_module):
        try:
            epoch = trainer.current_epoch
            out_dir = os.path.join(self.output_dir, f"epoch-{epoch}")
            os.makedirs(out_dir, exist_ok=True)
            
            # Save adapters (PEFT) and tokenizer
            try:
                pl_module.model.save_pretrained(out_dir)
            except Exception:
                # If model is wrapped, try accessing underlying model
                try:
                    pl_module.model.module.save_pretrained(out_dir)
                except Exception:
                    pass
            try:
                tokenizer.save_pretrained(out_dir)
            except Exception:
                pass
            
            # Track saved epochs
            self.saved_epochs.append(epoch)
            
            # Remove old checkpoints if num_checkpoint is set
            if self.num_checkpoint > 0 and len(self.saved_epochs) > self.num_checkpoint:
                epochs_to_remove = self.saved_epochs[:-self.num_checkpoint]
                for old_epoch in epochs_to_remove:
                    old_dir = os.path.join(self.output_dir, f"epoch-{old_epoch}")
                    if os.path.exists(old_dir):
                        import shutil
                        shutil.rmtree(old_dir)
                        print(f"Removed old checkpoint: {old_dir}")
                self.saved_epochs = self.saved_epochs[-self.num_checkpoint:]
        except Exception as e:
            print(f"Error saving checkpoint at epoch {epoch}: {e}")