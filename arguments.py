from dataclasses import dataclass, field 
from typing import List 

@dataclass
class TrainingArguments: 
    project_name: str = field(
        default='dynamic-document-embedding',
        metadata={'help': 'Name of the W&B project'}
    )
    report_name: str = field(
        default='training_run',
        metadata={'help': 'Name of the W&B report and model save directory'}
    )
    num_epochs: int = field(
        default=5,
        metadata={"help": "Number of epochs to train for."}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Learning rate for the optimizer."}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for training."}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for tokenization."}
    )
    datasets: List[str] = field(
        default_factory=lambda: ["squad"],
        metadata={"help": "List of datasets to use for training."}
    )
    use_dual_loss: bool = field(
        default=True,
        metadata={"help": "Whether to use both contrastive and entropy losses."}
    )
    entropy_weight: float = field(
        default=0.01,
        metadata={"help": "Weight for the entropy loss term."}
    )
    pretrained_model_path: str = field(
        default="dmis-lab/biobert-v1.1",
        metadata={"help": "Training initial checkpoint."}
    )
    model_save_path: str = field(
        default="/fs/ess/PAS0027/Rui-research/PAM_Checkpoint",
        metadata={"help": "Directory to save the trained model."}
    )