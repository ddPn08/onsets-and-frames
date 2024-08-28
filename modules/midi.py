from dataclasses import dataclass
from typing import List, Optional

import pretty_midi as pm
import torch

from modules.constants import (
    HOP_LENGTH,
    HOPS_IN_OFFSET,
    HOPS_IN_ONSET,
    MAX_MIDI,
    MIN_MIDI,
    SAMPLE_RATE,
)


@dataclass
class Note:
    pitch: int
    start: float
    end: float
    velocity: int


@dataclass
class Pedal:
    start: float
    end: float


def parse_midi(midi_path: str):
    midi = pm.PrettyMIDI(midi_path)

    notes: List[Note] = []
    pedals: List[Pedal] = []

    for note in midi.instruments[0].notes:
        notes.append(Note(note.pitch, note.start, note.end, note.velocity))

    pedal: Optional[Pedal] = None

    for cc in midi.instruments[0].control_changes:
        if cc.number == 64:
            if cc.value > 64:
                if pedal is None:
                    pedal = Pedal(cc.time, None)
            elif pedal is not None:
                pedal.end = cc.time
                pedals.append(pedal)
                break
            elif len(pedals) > 0:
                pedals[-1].end = cc.time

    return notes, pedals


def label_events(notes: List[Note], pedals: List[Pedal], audio_length: int):
    n_keys = MAX_MIDI - MIN_MIDI + 1
    n_steps = (audio_length - 1) // HOP_LENGTH + 1

    note_label = torch.zeros((n_steps, n_keys), dtype=torch.uint8)
    velocity = torch.zeros((n_steps, n_keys), dtype=torch.uint8)

    pedal_label = torch.zeros(n_steps, dtype=torch.uint8)

    for note in notes:
        left = int(round(note.start * SAMPLE_RATE / HOP_LENGTH))
        onset_right = min(n_steps, left + HOPS_IN_ONSET)
        frame_right = int(round(note.end * SAMPLE_RATE / HOP_LENGTH))
        frame_right = min(n_steps, frame_right)
        offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

        f = int(note.pitch) - MIN_MIDI
        note_label[left:onset_right, f] = 3
        note_label[onset_right:frame_right, f] = 2
        note_label[frame_right:offset_right, f] = 1
        velocity[left:frame_right, f] = note.velocity

    for pedal in pedals:
        left = int(round(pedal.start * SAMPLE_RATE / HOP_LENGTH))
        onset_right = min(n_steps, left + HOPS_IN_ONSET)
        frame_right = int(round(pedal.end * SAMPLE_RATE / HOP_LENGTH))
        frame_right = min(n_steps, frame_right)
        offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

        pedal_label[left:onset_right] = 3
        pedal_label[onset_right:frame_right] = 2
        pedal_label[frame_right:offset_right] = 1

    return note_label, velocity, pedal_label


def create_midi(
    notes: List[Note],
):
    midi = pm.PrettyMIDI()
    instrument = pm.Instrument(0)

    for note in notes:
        instrument.notes.append(
            pm.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start,
                end=note.end,
            )
        )

    midi.instruments.append(instrument)

    return midi
