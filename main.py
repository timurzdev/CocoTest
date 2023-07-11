import argparse
import os
from typing import Optional

from datamodule import CocoDataModule
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from model import FasterRCNNModule


def train(
    logger_name: str,
    data_folder: str,
    batch_size: int,
    epochs: int,
    ckpt_path: Optional[str] = None,
    pretrained: bool = False,
):
    logger = WandbLogger(logger_name)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=2,
                verbose=False,
                mode="min",
            )
        ],
        logger=logger,
    )
    if pretrained:
        module = FasterRCNNModule.load_from_checkpoint(ckpt_path)
    else:
        module = FasterRCNNModule(
            91, 0.45, os.path.join(data_folder, "annotations", "instances_val2017.json")
        )

    data_module = CocoDataModule(data_folder, batch_size=batch_size)
    module.train()
    trainer.fit(module, data_module)
    trainer.save_checkpoint(f"./checkpoints/{logger_name}_best.ckpt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--chkpt_path", type=str, default="./checkpoints/best.ckpt")
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--logger_name", type=str, default="default_logger")
    parser.add_argument("--data_folder", type=str, default="./data/coco2017")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.train:
        train(
            args.logger_name,
            args.data_folder,
            args.batch_size,
            args.epochs,
            args.chkpt_path,
            args.pretrained,
        )


if __name__ == "__main__":
    main()
