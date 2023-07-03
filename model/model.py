import lightning.pytorch as pl
import torch
import torch.optim as optim
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def get_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


#
# def convert_to_targets(annotations):
#     targets = []
#     for image_annotations in annotations:
#         target = {}
#         boxes = []
#         labels = []
#         for annotation in image_annotations:
#             x_min, y_min, width, height = annotation["bbox"]
#             boxes.append([x_min, y_min, x_min + width, y_min + height])
#             labels.append(annotation["category_id"])
#         target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
#         target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
#         targets.append(target)
#     return targets
#

class FasterRCNNModule(pl.LightningModule):
    def __init__(self, num_classes: int, iou_threshold: float, annotation_path: str, lr: float = 1e-4):
        super().__init__()
        self.model = get_model(num_classes)
        self.lr = lr
        self.iou_threshold = iou_threshold
        self.val_annotation_path = annotation_path

        # Save hyperparameters
        # Saves model arguments to the ``hparams`` attribute.
        self.save_hyperparameters()
        self.cocoGt = COCO(self.val_annotation_path)  # replace with path to your annotation file
        self.results = []

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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.model.train()
        x, y = batch
        loss_dict = self.model(x, y)
        loss = sum(loss for loss in loss_dict.values())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.train()
        x, y = batch
        with torch.no_grad():
            loss_dict = self.model(x, y)
        loss = sum(loss for loss in loss_dict.values())
        outputs = self(x)
        for output, target in zip(outputs, y):
            self.results.extend(self.format_for_evaluation(output, target))
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        cocoDt = self.cocoGt.loadRes(self.results)
        cocoEval = COCOeval(self.cocoGt, cocoDt, iouType='bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        # Get the Average Precision (AP) score
        avg_precision = cocoEval.stats[0]
        self.log('val_mAP', avg_precision)
