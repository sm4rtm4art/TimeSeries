from pytorch_lightning.callbacks import Callback


class PrintEpochResults(Callback):
    """Custom callback to update Streamlit progress bar during training."""

    def __init__(self, progress_bar, status_text, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        """Update progress bar and status text at the end of each epoch."""
        current_epoch = trainer.current_epoch
        progress = (current_epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(f"Training progress: {int(progress * 100)}%")
