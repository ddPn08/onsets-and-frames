from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
from lightning.pytorch import LightningModule

from modules.constants import (
    HOP_LENGTH,
    MEL_FMAX,
    MEL_FMIN,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_LENGTH,
)
from modules.evaluate import evaluate_note, evaluate_pedal
from modules.models import OnsetsAndFrames, OnsetsAndFramesPedal


def weighted_mse_loss(
    velocity_pred: torch.Tensor, velocity_label: torch.Tensor, onset_label: torch.Tensor
):
    denominator = onset_label.sum()
    if denominator.item() == 0:
        return denominator
    else:
        return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator


class TranscriberModule(LightningModule):
    def __init__(
        self,
        model,
        optimizer_class: Any,
        lr: float = 1e-4,
    ):
        super().__init__()
        if isinstance(model, OnsetsAndFrames):
            self.mode = "note"
        elif isinstance(model, OnsetsAndFramesPedal):
            self.mode = "pedal"
        else:
            raise ValueError("Invalid model type")

        self.model = model
        self.optimizer_class = optimizer_class
        self.lr = lr

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=WINDOW_LENGTH,
            win_length=WINDOW_LENGTH,
            hop_length=HOP_LENGTH,
            f_min=MEL_FMIN,
            f_max=MEL_FMAX,
            norm="slaney",
            power=1.0,
        ).to(self.device)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def note_training_step(self, batch: torch.Tensor, _: int):
        audio, onset_label, offset_label, frame_label, velocity_label = batch

        mel = self.mel_transform(audio.reshape(-1, audio.shape[-1])[:, :-1])
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = mel.transpose(-1, -2)

        onset_pred, offset_pred, _, frame_pred, velocity_pred = self.model(mel)

        onset_pred = onset_pred.reshape(onset_label.shape)
        offset_pred = offset_pred.reshape(offset_label.shape)
        frame_pred = frame_pred.reshape(frame_label.shape)
        velocity_pred = velocity_pred.reshape(velocity_label.shape)

        onset_loss = F.binary_cross_entropy_with_logits(onset_pred, onset_label)
        offset_loss = F.binary_cross_entropy_with_logits(offset_pred, offset_label)
        frame_loss = F.binary_cross_entropy_with_logits(frame_pred, frame_label)
        velocity_loss = weighted_mse_loss(velocity_pred, velocity_label, onset_label)

        loss = onset_loss + offset_loss + frame_loss + velocity_loss

        self.log("loss/onset", onset_loss)
        self.log("loss/offset", offset_loss)
        self.log("loss/frame", frame_loss)
        self.log("loss/velocity", velocity_loss)
        self.log("loss/total", loss, prog_bar=True)
        self.log(
            "loss/epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def pedal_training_step(self, batch: torch.Tensor, _: int):
        audio, onset_label, offset_label, frame_label = batch

        mel = self.mel_transform(audio.reshape(-1, audio.shape[-1])[:, :-1])
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = mel.transpose(-1, -2)

        onset_pred, offset_pred, _, frame_pred = self.model(mel)

        onset_pred = onset_pred.reshape(onset_label.shape)
        offset_pred = offset_pred.reshape(offset_label.shape)
        frame_pred = frame_pred.reshape(frame_label.shape)

        onset_loss = F.binary_cross_entropy_with_logits(onset_pred, onset_label)
        offset_loss = F.binary_cross_entropy_with_logits(offset_pred, offset_label)
        frame_loss = F.binary_cross_entropy_with_logits(frame_pred, frame_label)

        loss = onset_loss + offset_loss + frame_loss

        self.log("loss/onset", onset_loss)
        self.log("loss/offset", offset_loss)
        self.log("loss/frame", frame_loss)
        self.log("loss/total", loss, prog_bar=True)
        self.log(
            "loss/epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch: torch.Tensor, _: int):
        if self.mode == "note":
            return self.note_training_step(batch, _)
        else:
            return self.pedal_training_step(batch, _)

    def note_validation_step(self, batch: torch.Tensor, _: int):
        audio, onset_label, offset_label, frame_label, velocity_label = batch

        mel = self.mel_transform(audio.reshape(-1, audio.shape[-1])[:, :-1])
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = mel.transpose(-1, -2)

        onset_pred, offset_pred, _, frame_pred, velocity_pred = self.model(mel)

        onset_pred = onset_pred.reshape(onset_label.shape)
        offset_pred = offset_pred.reshape(offset_label.shape)
        frame_pred = frame_pred.reshape(frame_label.shape)
        velocity_pred = velocity_pred.reshape(velocity_label.shape)

        onset_loss = F.binary_cross_entropy_with_logits(onset_pred, onset_label)
        offset_loss = F.binary_cross_entropy_with_logits(offset_pred, offset_label)
        frame_loss = F.binary_cross_entropy_with_logits(frame_pred, frame_label)
        velocity_loss = weighted_mse_loss(velocity_pred, velocity_label, onset_label)

        onset_pred = onset_pred.sigmoid()
        offset_pred = offset_pred.sigmoid()
        frame_pred = frame_pred.sigmoid()

        for i in range(onset_label.shape[0]):
            metrics = evaluate_note(
                onset_label[i],
                frame_label[i],
                velocity_label[i],
                onset_pred[i],
                frame_pred[i],
                velocity_pred[i],
            )

            for key, value in metrics.items():
                self.log(f"val/{key}", value, on_epoch=True, sync_dist=True)

        loss = onset_loss + offset_loss + frame_loss + velocity_loss

        self.log("val/loss/onset", onset_loss, on_epoch=True, sync_dist=True)
        self.log("val/loss/offset", offset_loss, on_epoch=True, sync_dist=True)
        self.log("val/loss/frame", frame_loss, on_epoch=True, sync_dist=True)
        self.log("val/loss/velocity", velocity_loss, on_epoch=True, sync_dist=True)
        self.log("val/loss/total", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def pedal_validation_step(self, batch: torch.Tensor, _: int):
        audio, onset_label, offset_label, frame_label = batch

        mel = self.mel_transform(audio.reshape(-1, audio.shape[-1])[:, :-1])
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = mel.transpose(-1, -2)

        onset_pred, offset_pred, _, frame_pred = self.model(mel)

        onset_pred = onset_pred.reshape(onset_label.shape)
        offset_pred = offset_pred.reshape(offset_label.shape)
        frame_pred = frame_pred.reshape(frame_label.shape)

        onset_loss = F.binary_cross_entropy_with_logits(onset_pred, onset_label)
        offset_loss = F.binary_cross_entropy_with_logits(offset_pred, offset_label)
        frame_loss = F.binary_cross_entropy_with_logits(frame_pred, frame_label)

        onset_pred = onset_pred.sigmoid()
        offset_pred = offset_pred.sigmoid()
        frame_pred = frame_pred.sigmoid()

        for i in range(onset_label.shape[0]):
            metrics = evaluate_pedal(
                onset_label[i],
                frame_label[i],
                onset_pred[i],
                frame_pred[i],
            )

            for key, value in metrics.items():
                self.log(f"val/{key}", value, on_epoch=True, sync_dist=True)

        loss = onset_loss + offset_loss + frame_loss

        self.log("val/loss/onset", onset_loss, on_epoch=True, sync_dist=True)
        self.log("val/loss/offset", offset_loss, on_epoch=True, sync_dist=True)
        self.log("val/loss/frame", frame_loss, on_epoch=True, sync_dist=True)
        self.log("val/loss/total", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch: torch.Tensor, _: int):
        if self.mode == "note":
            return self.note_validation_step(batch, _)
        else:
            return self.pedal_validation_step(batch, _)

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.lr)
