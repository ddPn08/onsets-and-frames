import sys
from collections import defaultdict

import numpy as np
import torch
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import (
    precision_recall_f1_overlap as evaluate_notes_with_velocity,
)
from mir_eval.util import midi_to_hz
from scipy.stats import hmean

from modules.constants import HOP_LENGTH, MIN_MIDI, SAMPLE_RATE
from modules.decoding import (
    extract_notes,
    extract_pedals,
    notes_to_frames,
    pedals_to_frames,
)

eps = sys.float_info.epsilon


def evaluate_note(
    onset_true: torch.Tensor,
    frame_true:torch.Tensor,
    velocity_true: torch.Tensor,
    onset_pred: torch.Tensor,
    frame_pred: torch.Tensor,
    velocity_pred: torch.Tensor,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
):
    metrics = {}

    p_ref, i_ref, v_ref = extract_notes(onset_true, frame_true, velocity_true)
    p_est, i_est, v_est = extract_notes(
        onset_pred,
        frame_pred,
        velocity_pred,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )

    t_ref, f_ref = notes_to_frames(p_ref, i_ref, frame_true.shape)
    t_est, f_est = notes_to_frames(p_est, i_est, frame_pred.shape)

    scaling = HOP_LENGTH / SAMPLE_RATE

    i_ref = (i_ref * scaling).reshape(-1, 2)
    p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

    t_ref = t_ref.astype(np.float64) * scaling
    f_ref = [
        np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_ref
    ]
    t_est = t_est.astype(np.float64) * scaling
    f_est = [
        np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est
    ]

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
    metrics["metric/note/precision"] = p
    metrics["metric/note/recall"] = r
    metrics["metric/note/f1"] = f
    metrics["metric/note/overlap"] = o

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
    metrics["metric/note-with-offsets/precision"] = p
    metrics["metric/note-with-offsets/recall"] = r
    metrics["metric/note-with-offsets/f1"] = f
    metrics["metric/note-with-offsets/overlap"] = o

    p, r, f, o = evaluate_notes_with_velocity(
        i_ref,
        p_ref,
        v_ref,
        i_est,
        p_est,
        v_est,
        offset_ratio=None,
        velocity_tolerance=0.1,
    )
    metrics["metric/note-with-velocity/precision"] = p
    metrics["metric/note-with-velocity/recall"] = r
    metrics["metric/note-with-velocity/f1"] = f
    metrics["metric/note-with-velocity/overlap"] = o

    p, r, f, o = evaluate_notes_with_velocity(
        i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1
    )
    metrics["metric/note-with-offsets-and-velocity/precision"] = p
    metrics["metric/note-with-offsets-and-velocity/recall"] = r
    metrics["metric/note-with-offsets-and-velocity/f1"] = f
    metrics["metric/note-with-offsets-and-velocity/overlap"] = o

    frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metrics["metric/frame/f1"] = (
        hmean([frame_metrics["Precision"] + eps, frame_metrics["Recall"] + eps]) - eps
    )

    for key, loss in frame_metrics.items():
        metrics["metric/frame/" + key.lower().replace(" ", "_")] = loss

    return metrics


def evaluate_pedal(
    onset_true: torch.Tensor,
    frame_true: torch.Tensor,
    onset_pred: torch.Tensor,
    frame_pred: torch.Tensor,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
):
    metrics = {}

    i_ref = extract_pedals(onset_true, frame_true)
    i_est = extract_pedals(
        onset_pred,
        frame_pred,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )

    t_ref, f_ref = pedals_to_frames(i_ref, frame_true.shape)
    t_est, f_est = pedals_to_frames(i_est, frame_pred.shape)

    scaling = HOP_LENGTH / SAMPLE_RATE

    i_ref = (i_ref * scaling).reshape(-1, 2)
    i_est = (i_est * scaling).reshape(-1, 2)

    t_ref = t_ref.astype(np.float64) * scaling
    t_est = t_est.astype(np.float64) * scaling

    frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metrics["metric/frame/f1"] = (
        hmean([frame_metrics["Precision"] + eps, frame_metrics["Recall"] + eps]) - eps
    )

    for key, loss in frame_metrics.items():
        metrics["metric/frame/" + key.lower().replace(" ", "_")] = loss

    return metrics


def evaluate(data, model, onset_threshold=0.5, frame_threshold=0.5, save_path=None):
    metrics = defaultdict(list)

    for label in data:
        pred, losses = model.run_on_batch(label)

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        for key, value in pred.items():
            value.squeeze_(0).relu_()

        p_ref, i_ref, v_ref = extract_notes(
            label["onset"], label["frame"], label["velocity"]
        )
        p_est, i_est, v_est = extract_notes(
            pred["onset"],
            pred["frame"],
            pred["velocity"],
            onset_threshold,
            frame_threshold,
        )

        t_ref, f_ref = notes_to_frames(p_ref, i_ref, label["frame"].shape)
        t_est, f_est = notes_to_frames(p_est, i_est, pred["frame"].shape)

        scaling = HOP_LENGTH / SAMPLE_RATE

        i_ref = (i_ref * scaling).reshape(-1, 2)
        p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        t_ref = t_ref.astype(np.float64) * scaling
        f_ref = [
            np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_ref
        ]
        t_est = t_est.astype(np.float64) * scaling
        f_est = [
            np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est
        ]

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
        metrics["metric/note/precision"].append(p)
        metrics["metric/note/recall"].append(r)
        metrics["metric/note/f1"].append(f)
        metrics["metric/note/overlap"].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        metrics["metric/note-with-offsets/precision"].append(p)
        metrics["metric/note-with-offsets/recall"].append(r)
        metrics["metric/note-with-offsets/f1"].append(f)
        metrics["metric/note-with-offsets/overlap"].append(o)

        p, r, f, o = evaluate_notes_with_velocity(
            i_ref,
            p_ref,
            v_ref,
            i_est,
            p_est,
            v_est,
            offset_ratio=None,
            velocity_tolerance=0.1,
        )
        metrics["metric/note-with-velocity/precision"].append(p)
        metrics["metric/note-with-velocity/recall"].append(r)
        metrics["metric/note-with-velocity/f1"].append(f)
        metrics["metric/note-with-velocity/overlap"].append(o)

        p, r, f, o = evaluate_notes_with_velocity(
            i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1
        )
        metrics["metric/note-with-offsets-and-velocity/precision"].append(p)
        metrics["metric/note-with-offsets-and-velocity/recall"].append(r)
        metrics["metric/note-with-offsets-and-velocity/f1"].append(f)
        metrics["metric/note-with-offsets-and-velocity/overlap"].append(o)

        frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        metrics["metric/frame/f1"].append(
            hmean([frame_metrics["Precision"] + eps, frame_metrics["Recall"] + eps])
            - eps
        )

        for key, loss in frame_metrics.items():
            metrics["metric/frame/" + key.lower().replace(" ", "_")].append(loss)

        # if save_path is not None:
        #     os.makedirs(save_path, exist_ok=True)
        #     label_path = os.path.join(save_path, os.path.basename(label['path']) + '.label.png')
        #     save_pianoroll(label_path, label['onset'], label['frame'])
        #     pred_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.png')
        #     save_pianoroll(pred_path, pred['onset'], pred['frame'])
        #     midi_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.mid')
        #     save_midi(midi_path, p_est, i_est, v_est)

    return metrics
