import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CocoDetection
from torchvision import transforms

import os


class CustomBatchs:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = transposed_data[1]

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        return (self.inp, self.tgt)


def collate_wrapper(batch):
    if torch.cuda.is_available():
        return CustomBatchs(batch)
    else:
        return tuple(zip(*batch))


class CocoDataModule(pl.LightningDataModule):
    def __init__(self,
                 base_data_folder: str,
                 batch_size: int,
                 ):
        super().__init__()
        self.base_path = base_data_folder
        self.batch_size = batch_size
        self.train_annotation_path = os.path.join(self.base_path, "annotations", "instances_train2017.json")
        self.train_data_path = os.path.join(self.base_path, "train2017")
        self.val_annotation_path = os.path.join(self.base_path, "annotations", "instances_val2017.json")
        self.val_data_path = os.path.join(self.base_path, "val2017")
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def prepare_data(self) -> None:
        dataset = CocoDetection(self.train_data_path, self.train_annotation_path, transform=self.transform)
        self.train_dataset, self.val_dataset = random_split(dataset, [0.8, 0.2])
        self.test_dataset = CocoDetection(self.val_data_path, self.val_annotation_path, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True,
                          collate_fn=collate_wrapper)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True,
                          collate_fn=collate_wrapper)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True,
                          collate_fn=collate_wrapper)
