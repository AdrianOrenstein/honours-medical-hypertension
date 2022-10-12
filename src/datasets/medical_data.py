from collections import OrderedDict
import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple

from loguru import logger
import numpy as np
import pytorch_lightning as pl
from src.datasets.utils import dfs_unpack_json
import torch

Json = Dict[str, Any]


@dataclasses.dataclass
class PatientData:
    uid: int
    label: int
    data: OrderedDict

    @property
    def human_readable_label(self) -> str:
        """
        Yes if the patient is hypertensive
        """
        if self.label == 1:
            return "Yes"
        elif self.label == 0:
            return "No"
        elif self.label == -1:
            return "Maybe"
        else:
            assert False, "Unknown label"


class PadData:
    def __init__(
        self,
        pad_to_length: int,
        pad_val: int,
    ):
        self.pad_to_length = pad_to_length
        self.pad_val = pad_val

    def __call__(self, data: List[Tuple[torch.LongTensor, torch.LongTensor]]):
        if len(data) == 2:
            sample, label, patient_id = data

            return (
                torch.cat(
                    (
                        sample,
                        torch.tensor(
                            [self.pad_val]
                            * ((self.pad_to_length or 2**14) - len(sample))
                        ).long(),
                    )
                )[: self.pad_to_length],
                label,
                patient_id,
            )
        else:
            samples: List[torch.LongTensor] = [X.flatten() for X, y, s in data]
            labels: List[torch.LongTensor] = [y for X, y, s in data]
            patient_ids: List[torch.LongTensor] = [s for X, y, s in data]

            padded_samples = [
                torch.cat(
                    (
                        sample,
                        torch.tensor(
                            [self.pad_val]
                            * ((self.pad_to_length or 2**14) - len(sample))
                        ).long(),
                    )
                )[: self.pad_to_length]
                for sample in samples
            ]

            return (
                torch.stack(padded_samples),
                torch.stack(labels).flatten(),
                torch.stack(patient_ids).flatten(),
            )


class PatientDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_directory_path: Path,
        preprocessing: Callable[[Json], List[int]],
        balance_classes: bool,
    ):
        """Initialization"""
        self.balance_classes = balance_classes
        self.class_folders = sorted(str(t) for t in data_directory_path.glob("*"))

        # remove ds_store
        self.class_folders = [
            p
            for p in self.class_folders
            if p.endswith("class0") or p.endswith("class1")
        ]

        self.class_lookup: Dict[str, int] = dict(
            zip(self.class_folders, range(len(self.class_folders)))
        )
        self.dataset_files: List[Path] = self.prep_data(data_directory_path)

        self.preprocessing = preprocessing

    def prep_data(self, path: Path) -> List[Path]:
        assert path.is_dir()

        classes = {}
        for class_filepath in self.class_folders:
            class_filepath = class_filepath.split("/")[-1]
            classes[class_filepath] = list(path.glob(class_filepath + "/*.pt"))
            logger.info(
                "  ".join(
                    [
                        str(path),
                        class_filepath + "/*.pt",
                        str(len(classes[class_filepath])),
                    ]
                )
            )

        min_number_of_samples_for_classes = float("inf")
        for class_name, class_files in classes.items():
            min_number_of_samples_for_classes = min(
                min_number_of_samples_for_classes, len(class_files)
            )

        final_dataset = []
        num_samples_per_class = []
        for class_name, class_files in classes.items():

            if self.balance_classes == "True":
                data_from_class = class_files[:min_number_of_samples_for_classes]
            else:
                data_from_class = class_files

            logger.info(f"{class_name} = {len(data_from_class)}")
            final_dataset.extend(data_from_class)

            num_samples_per_class.append(len(data_from_class))

        num_samples_per_class = np.array(num_samples_per_class)
        weight_per_class = num_samples_per_class.sum() / num_samples_per_class

        weight_per_class /= weight_per_class.sum()

        self.weight_per_class = torch.Tensor(weight_per_class)

        return final_dataset

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.dataset_files)

    def get_sample(self, filename: Path) -> PatientData:
        filename_str = str(filename)
        parent_dir = str(filename.parent)

        patient_data: PatientData = torch.load(filename_str)

        if not patient_data:
            logger.warning(
                f"No json extracted from {repr(patient_data.data)} at {filename}"
            )

        return patient_data

    def __getitem__(
        self, index
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Generates one sample of data"""
        # Load data and get label
        filename = self.dataset_files[index]
        data = self.get_sample(filename)

        # torch.save(data, "test_data.pt")
        # torch.save(filename, "filename.pt")

        # print("saved")

        # return None

        patient_data = self.preprocessing(data.data)
        patient_label = torch.LongTensor([self.class_lookup[str(filename.parent)]])
        patient_uid = torch.LongTensor([data.uid])

        return patient_data, patient_label, patient_uid


class MedicalDatasetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_dir: Path,
        val_data_dir: Path,
        preprocessing: Callable[[Json], List[int]],
        pad_token_id: int,
        batch_size: int,
        sequence_length: int,
        balance_classes: bool,
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.preprocessing = preprocessing
        self.pad_token_id = pad_token_id
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.balance_classes = balance_classes
        self.setup()

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = PatientDataset(
            data_directory_path=self.train_data_dir,
            preprocessing=self.preprocessing,
            balance_classes=self.balance_classes,
        )
        self.val_dataset = PatientDataset(
            data_directory_path=self.val_data_dir,
            preprocessing=self.preprocessing,
            balance_classes=True,
        )

        self.test_dataset = PatientDataset(
            data_directory_path=self.val_data_dir,
            preprocessing=self.preprocessing,
            balance_classes=False,
        )

        self.num_classes = len(self.train_dataset.class_folders)
        assert len(self.train_dataset.class_folders) == len(
            self.val_dataset.class_folders
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=PadData(
                pad_to_length=self.sequence_length,
                pad_val=self.pad_token_id,
            ),
            num_workers=4,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=PadData(
                pad_to_length=self.sequence_length,
                pad_val=self.pad_token_id,
            ),
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=PadData(
                pad_to_length=self.sequence_length,
                pad_val=self.pad_token_id,
            ),
            num_workers=4,
            pin_memory=True,
        )


def parse_json_into_only_fields(patient_data: Json) -> Iterable[Tuple[str, str]]:
    for dataset_name, patient_dataset in patient_data.items():
        if not patient_dataset:
            continue
        for time, patient_fields_at_time in patient_dataset.items():
            if not patient_fields_at_time:
                continue
            for patient_field in patient_fields_at_time:
                for k_v_entry in patient_field.items():
                    yield k_v_entry


# if __name__ == "__main__":
#     patient_data = torch.load('test_data.pt')
#     filename = torch.load('filename.pt')

#     logger.info(f"{patient_data.uid=}, {patient_data.label=}")


#     flattened_json_input: List[Tuple[str, str]] = list(
#         parse_json_into_only_fields(patient_data.data)
#     )

#     print(flattened_json_input)
