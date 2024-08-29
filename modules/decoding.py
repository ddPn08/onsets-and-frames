from typing import List

import numpy as np
import torch

from modules.constants import MIN_MIDI
from modules.midi import Note, Pedal


def extract_notes(
    onsets: torch.Tensor,
    frames: torch.Tensor,
    velocity: torch.Tensor,
    scaling: float,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
):
    onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
    frames = (frames > frame_threshold).cpu().to(torch.uint8)
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

    pitches = []
    intervals = []
    velocities = []

    notes: List[Note] = []

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            if onsets[offset, pitch].item():
                velocity_samples.append(velocity[offset, pitch].item())
            offset += 1
            if offset == onsets.shape[0]:
                break

        if offset > onset:
            note = Note(
                MIN_MIDI + pitch,
                start=onset * scaling,
                end=offset * scaling,
                velocity=min(127, int(np.mean(velocity_samples) * 127 if len(velocity_samples) > 0 else 0)),
            )
            notes.append(note)
            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(
                np.mean(velocity_samples) if len(velocity_samples) > 0 else 0
            )

    return notes

def extract_pedals(
    onsets: torch.Tensor,
    frames: torch.Tensor,
    scaling: float,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
):
    onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
    frames = (frames > frame_threshold).cpu().to(torch.uint8)
    onset_diff = torch.cat([onsets[:1], onsets[1:] - onsets[:-1]], dim=0) == 1

    pedals: List[Pedal] = []

    for i, pedal_on in enumerate(onset_diff):
        if pedal_on.item():
            onset = i
            offset = i
            while frames[offset].item():
                offset += 1
                if offset == frames.shape[0]:
                    break
            pedal = Pedal(
                start=onset * scaling,
                end=offset * scaling,
            )
            pedals.append(pedal)

    return pedals