import argparse

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torchvision import transforms

from model import FasterRCNNModule
from datamodule import CocoDataModule


def train(logger_name: str, data_folder: str, batch_size: int, epochs: int):
    logger = WandbLogger(logger_name)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        devices=1,
        callbacks=[EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=2,
            verbose=False,
            mode="min",
        )],
        logger=logger
    )
    module = FasterRCNNModule(80, 0.45)
    data_module = CocoDataModule(data_folder, batch_size=batch_size)
    module.train()
    trainer.fit(module, data_module)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--predict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--logger_name", type=str, default="default_logger")
    parser.add_argument("--data_folder", type=str, default="./data/coco2017")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.train:
        train(args.logger_name, args.data_folder, args.batch_size, args.epochs)


if __name__ == '__main__':
    main()
