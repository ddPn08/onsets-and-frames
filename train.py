import os
from typing import Literal

import fire
import torch
import torch.utils.data as data
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

from modules.constants import MAX_MIDI, MIN_MIDI, N_MELS
from modules.dataset import MaestroDataset
from modules.models import OnsetsAndFrames, OnsetsAndFramesPedal
from modules.training import TranscriberModule

torch.set_float32_matmul_precision("medium")


def main(
    dataset_dir: str = "./maestro-v3.0.0-preprocessed",
    output_dir: str = "./output",
    mode: Literal["note", "pedal"] = "note",
    accelerator: str = "gpu",
    devices: str = "0,",
    max_train_epochs: int = 100,
    precision: _PRECISION_INPUT = 32,
    batch_size: int = 1,
    sequence_length: int = 327680,
    model_complexity: int = 48,
    num_workers: int = 1,
    optimizer: str = "adam",
    learning_rate: float = 1e-4,
    logger: str = "none",
    logger_name: str = "training",
    logger_project: str = "onsets-and-frames",
):
    dataset = MaestroDataset(
        dataset_dir, "train", sequence_length=sequence_length, mode=mode
    )
    val_dataset = MaestroDataset(
        dataset_dir, "validation", sequence_length=sequence_length, mode=mode
    )

    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )

    model = (
        OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity)
        if mode == "note"
        else OnsetsAndFramesPedal(N_MELS, model_complexity)
    )

    if optimizer == "adam":
        optimizer_class = torch.optim.Adam
    else:
        raise ValueError("Invalid optimizer")

    module = TranscriberModule(model, optimizer_class, learning_rate)

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    callbacks = [
        ModelCheckpoint(
            every_n_train_steps=1000,
            dirpath=checkpoint_dir,
            save_top_k=10,
            mode="max",
            monitor="step",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if logger == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(
            name=logger_name,
            project=logger_project,
        )
    elif logger == "tensorboard":
        from pytorch_lightning.loggers import TensorBoardLogger

        logger = TensorBoardLogger("lightning_logs", name=logger_name)
    else:
        logger = None

    trainer = Trainer(
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_train_epochs,
        log_every_n_steps=1,
        callbacks=callbacks,
        precision=precision,
    )
    trainer.fit(module, dataloader, val_dataloader)

    state_dict = module.model.state_dict()
    torch.save(state_dict, os.path.join(output_dir, "model.pt"))


if __name__ == "__main__":
    fire.Fire(main)
