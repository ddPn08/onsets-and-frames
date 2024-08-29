import json
import os
from typing import List, Literal, Tuple

import torch
import torch.utils.data as data
from pydantic import BaseModel

from modules.constants import HOP_LENGTH, SAMPLE_RATE


class Metadata(BaseModel):
    canonical_composer: str
    canonical_title: str
    split: str
    year: int
    midi_filename: str
    audio_filename: str
    duration: float


class Segment(BaseModel):
    audio: str
    label: str
    onset: int
    offset: int


class TranscriptionDataset(data.Dataset):
    def __init__(
        self,
        files: List[Tuple[str, str, int]],
        sequence_length: int = 327680,
        mode: Literal["note", "pedal"] = "note",
    ):
        self.files = files
        self.segments: List[Segment] = []
        self.sequence_length = sequence_length
        self.mode = mode

        for audio, label, duration in self.files:
            num_frame = int(duration * SAMPLE_RATE)
            for i in range(0, num_frame, sequence_length):
                self.segments.append(
                    Segment(
                        audio=audio,
                        label=label,
                        onset=i,
                        offset=min(i + sequence_length, num_frame),
                    )
                )

    def __len__(self):
        return len(self.segments)

    def get_item_note(self, idx: int):
        segment = self.segments[idx]

        audio = torch.load(segment.audio, weights_only=True)  # (T,)
        label = torch.load(segment.label, weights_only=True)

        note_label = label["note"]  # (T, 88)
        velocity = label["velocity"]  # (T, 88)

        start_step = segment.onset // HOP_LENGTH
        end_step = segment.offset // HOP_LENGTH

        audio = audio[segment.onset : segment.offset]
        note_label = note_label[start_step:end_step]
        velocity = velocity[start_step:end_step]

        sequence_steps = self.sequence_length // HOP_LENGTH

        padding = self.sequence_length - len(audio)
        padding_step = sequence_steps - len(note_label)
        pad_audio = torch.zeros(padding, dtype=audio.dtype, device=audio.device)
        pad_label = torch.zeros(
            (padding_step, note_label.shape[1]),
            dtype=note_label.dtype,
            device=note_label.device,
        )

        audio = torch.cat([audio, pad_audio])
        note_label = torch.cat([note_label, pad_label])
        velocity = torch.cat([velocity, pad_label])

        onset = (note_label == 3).float()
        offset = (note_label == 1).float()
        frame = (note_label > 1).float()
        velocity = velocity.float() / 127.0

        return audio, onset, offset, frame, velocity

    def get_item_pedal(self, idx: int):
        segment = self.segments[idx]

        audio = torch.load(segment.audio, weights_only=True)
        label = torch.load(segment.label, weights_only=True)

        pedal_label = label["pedal"]  # (T,)

        start_step = segment.onset // HOP_LENGTH
        end_step = segment.offset // HOP_LENGTH

        audio = audio[segment.onset : segment.offset]
        pedal_label = pedal_label[start_step:end_step]

        sequence_steps = self.sequence_length // HOP_LENGTH

        padding = self.sequence_length - len(audio)
        padding_step = sequence_steps - len(pedal_label)

        pad_audio = torch.zeros(padding, dtype=audio.dtype, device=audio.device)
        pad_label = torch.zeros(
            padding_step, dtype=pedal_label.dtype, device=pedal_label.device
        )

        audio = torch.cat([audio, pad_audio])
        pedal_label = torch.cat([pedal_label, pad_label])

        onset = (pedal_label == 3).float()
        offset = (pedal_label == 1).float()

        return audio, onset, offset

    def __getitem__(self, idx: int):
        if self.mode == "note":
            return self.get_item_note(idx)
        else:
            return self.get_item_pedal(idx)

    def note_collate_fn(self, batch: torch.Tensor):
        audio, onset, offset, frame, velocity = zip(*batch)
        audio = torch.stack(audio)
        onset = torch.stack(onset)
        offset = torch.stack(offset)
        frame = torch.stack(frame)
        velocity = torch.stack(velocity)
        return audio, onset, offset, frame, velocity

    def pedal_collate_fn(self, batch: torch.Tensor):
        audio, onset, offset= zip(*batch)
        audio = torch.stack(audio)
        onset = torch.stack(onset)
        offset = torch.stack(offset)
        return audio, onset, offset

    def collate_fn(self, batch: torch.Tensor):
        if self.mode == "note":
            return self.note_collate_fn(batch)
        else:
            return self.pedal_collate_fn(batch)


class MaestroDataset(TranscriptionDataset):
    def __init__(
        self,
        dataset_dir: str,
        split: Literal["train", "validation", "test"] = "train",
        sequence_length: int = 327680,
        mode: Literal["note", "pedal"] = "note",
    ):
        with open(os.path.join(dataset_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            metadata = [Metadata(**m) for m in metadata]

        metadata = [m for m in metadata if m.split == split]

        files = []
        for m in metadata:
            audio = os.path.join(
                dataset_dir,
                "wav",
                m.split,
                m.audio_filename.replace("/", "-").replace("wav", "pt"),
            )
            label = os.path.join(
                dataset_dir,
                "label",
                m.split,
                m.midi_filename.replace("/", "-").replace("midi", "pt"),
            )
            files.append((audio, label, m.duration))

        super().__init__(files, sequence_length, mode)
