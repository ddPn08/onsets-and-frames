from typing import List

import numpy as np
import torch

from modules.constants import MIN_MIDI
from modules.midi import Note


def extract_notes(
    onsets: torch.Tensor,
    frames: torch.Tensor,
    velocity: torch.Tensor,
    scaling: float,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
):
    """
    Finds the note timings based on the onsets and frames information

    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float

    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
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
                velocity=int(np.mean(velocity_samples))
                if len(velocity_samples) > 0
                else 0,
            )
            notes.append(note)
            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(
                np.mean(velocity_samples) if len(velocity_samples) > 0 else 0
            )

    return notes
