import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CocoDetection
from torchvision import transforms

import os


class CustomCocoDetection(CocoDetection):
    def __init__(self, root, annFile, image_size, transform=None):
        super().__init__(root, annFile)
        self.transform = transform
        self.image_size = image_size  # size the images will be resized to

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        # Resize and adjust bounding boxes
        boxes = []
        labels = []
        for t in target:
            x_min, y_min, width, height = t['bbox']
            # Adjust bounding boxes to match resized image
            boxes.append([
                x_min * self.image_size[0] / img.width,
                y_min * self.image_size[1] / img.height,
                (x_min + width) * self.image_size[0] / img.width,
                (y_min + height) * self.image_size[1] / img.height
            ])
            labels.append(t['category_id'])

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def collate_fn(batch):
    images, targets = zip(*batch)

    # Assume that each image in the batch has the same size
    # so we can use a simple collation function.
    images = torch.stack(images)

    return images, targets


class CocoDataModule(pl.LightningDataModule):
    def __init__(self,
                 base_data_folder: str,
                 batch_size: int,
                 image_size: (int, int) = (256, 256)
                 ):
        super().__init__()
        self.base_path = base_data_folder
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_annotation_path = os.path.join(self.base_path, "annotations", "instances_train2017.json")
        self.train_data_path = os.path.join(self.base_path, "train2017")
        self.val_annotation_path = os.path.join(self.base_path, "annotations", "instances_val2017.json")
        self.val_data_path = os.path.join(self.base_path, "val2017")
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(image_size),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def prepare_data(self) -> None:
        dataset = CustomCocoDetection(self.train_data_path, self.train_annotation_path, image_size=self.image_size,
                                      transform=self.transform)
        self.train_dataset, self.val_dataset = random_split(dataset, [0.8, 0.2])
        self.test_dataset = CustomCocoDetection(self.val_data_path, self.val_annotation_path,
                                                image_size=self.image_size, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True,
                          collate_fn=collate_fn)
