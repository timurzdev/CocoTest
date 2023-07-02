import lightning.pytorch as pl
import torch
import torch.optim as optim
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def convert_to_targets(annotations):
    targets = []
    for image_annotations in annotations:
        target = {}
        boxes = []
        labels = []
        for annotation in image_annotations:
            x_min, y_min, width, height = annotation['bbox']
            boxes.append([x_min, y_min, x_min + width, y_min + height])
            labels.append(annotation['category_id'])
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        targets.append(target)
    return targets


class FasterRCNNModule(pl.LightningModule):
    def __init__(self, num_classes: int,
                 iou_threshold: float,
                 lr: float = 1e-4):
        super().__init__()
        self.model = get_model(num_classes)
        self.lr = lr
        self.iou_threshold = iou_threshold

        # Save hyperparameters
        # Saves model arguments to the ``hparams`` attribute.
        self.save_hyperparameters()

        # outputs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            threshold=0.0001,
            threshold_mode="abs",
        )

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            },
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = convert_to_targets(y)
        loss_dict = self.model(x, y)
        loss = sum(loss for loss in loss_dict.values())
        self.log_dict(loss_dict)
        return loss
