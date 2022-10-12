import copy
from pathlib import Path
import string
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from src.datasets.medical_data import MedicalDatasetDataModule
from src.datasets.utils import DESIGN_SUMMARY_SCHEMA_JSON
from src.experiments.base import BaseExperiment
from src.models.resnet_key_value import JSONKeyValueResnet_EmbeddingEncoder
import torch
import torchmetrics as pl_metrics


class ClassificationExperiment(BaseExperiment):
    """
    Relevant hyperparameters:
        https://github.com/p-lambda/wilds/blob/main/examples/configs/datasets.py#L136

    """

    NAME = "resnet"
    TAGS = {
        "MLFLOW_RUN_NAME": NAME,
        "dataset": "medical-hypertension",
        "algorithm": "resnet",
        "model": "torchvision.models.resnet50_1D",
    }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group(ClassificationExperiment.NAME)
        parser.add_argument("--learning_rate", type=float, default=0.003)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--sequence_length", type=int, default=16384)

        parser.add_argument(
            "--train_data_path",
            type=str,
            default=".data/medical_data_train_with_metadata",
        )
        parser.add_argument(
            "--val_data_path", type=str, default=".data/medical_data_val_with_metadata"
        )

        return parent_parser

    def __init__(
        self,
        learning_rate: float,
        batch_size: int,
        sequence_length: int,
        train_data_path: Path,
        val_data_path: Path,
        balanced_dataset: bool,
        metrics: dict = None,
        **kwargs: Optional[Any],
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path

        self.balanced_dataloaders = balanced_dataset

        self.metrics: Dict[str, Callable] = metrics or {
            "Accuracy": pl_metrics.classification.Accuracy(),
            "Precision": pl_metrics.classification.Precision(),
            "Recall": pl_metrics.classification.Recall(),
        }

        self.schema = self.parse_design_summary_schema(DESIGN_SUMMARY_SCHEMA_JSON)

        self.model = JSONKeyValueResnet_EmbeddingEncoder(
            num_classes=2,
            vocab=list(self.schema) + list(string.printable),
            layers=[3, 4, 6, 3],
        )
        self.data_module = MedicalDatasetDataModule(
            train_data_dir=Path(self.train_data_path),
            val_data_dir=Path(self.val_data_path),
            preprocessing=copy.deepcopy(self.model.preprocessing),
            pad_token_id=copy.deepcopy(self.model.tokeniser.pad_token_id),
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            balance_classes=self.balanced_dataloaders,
        )

        logger.warning(
            f"using train_dataset.weight_per_class={self.data_module.train_dataset.weight_per_class}"
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.data_module.train_dataset.weight_per_class
        )

    def calculate_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_hat, y)

    def parse_design_summary_schema(self, schema_json: Dict[str, Any]) -> List[str]:
        return schema_json["schema"]

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-06)

        # sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     opt, factor=0.1, min_lr=1e-6, verbose=True
        # )

        return {
            "optimizer": opt,
            # "lr_scheduler": sch,
            # "monitor": "val_loss",
            # "interval": "epoch",
            # "frequency": 2,
        }
