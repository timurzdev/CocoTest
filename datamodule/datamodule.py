import os

import lightning.pytorch as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CocoDetection


class CustomCocoDetection(CocoDetection):
    def __init__(self, root, annFile, image_size, transform=None):
        super().__init__(root, annFile)
        self.transform = transform
        self.image_size = image_size  # size the images will be resized to

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img_id = self.ids[idx]
        boxes = []
        labels = []
        for t in target:
            x_min, y_min, width, height = t["bbox"]
            # Adjust bounding boxes to match resized image
            boxes.append(
                [
                    x_min * self.image_size[0] / img.width,
                    y_min * self.image_size[1] / img.height,
                    (x_min + width) * self.image_size[0] / img.width,
                    (y_min + height) * self.image_size[1] / img.height,
                ]
            )
            labels.append(t["category_id"])

        target_new = {}
        target_new["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target_new["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target_new["image_id"] = torch.tensor([img_id], dtype=torch.int64)

        if self.transform is not None:
            img = self.transform(img)
        print("________", target_new, "_______")
        return img, target_new


def collate_fn(batch):
    images = [item[0] for item in batch]
    boxes = [item[1]["boxes"] for item in batch]
    labels = [item[1]["labels"] for item in batch]
    image_ids = [item[1]["image_id"] for item in batch]

    images = torch.stack(images)
    boxes = pad_sequence(boxes, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)
    image_ids = torch.stack(image_ids)

    targets = {"boxes": boxes, "labels": labels, "image_id": image_ids}

    return images, targets


class CocoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        base_data_folder: str,
        batch_size: int,
        image_size: (int, int) = (256, 256),
    ):
        super().__init__()
        self.base_path = base_data_folder
        self.image_size = image_size
        self.batch_size = batch_size
        self.annotation_path = os.path.join(
            self.base_path, "annotations", "instances_val2017.json"
        )
        self.data_path = os.path.join(self.base_path, "images", "val2017")
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
        dataset = CustomCocoDetection(
            self.data_path,
            self.annotation_path,
            image_size=self.image_size,
            transform=self.transform,
        )
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [0.7, 0.15, 0.15]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            collate_fn=collate_fn,
        )
