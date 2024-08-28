import fire
import torch
import torchaudio
import tqdm

from modules.audio import load_audio
from modules.constants import (
    HOP_LENGTH,
    MAX_MIDI,
    MEL_FMAX,
    MEL_FMIN,
    MIN_MIDI,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_LENGTH,
)
from modules.decoding import extract_notes
from modules.midi import create_midi
from modules.models import OnsetsAndFrames
from modules.utils import save_pianoroll


def fix_state_dict(state_dict):
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any(k.startswith("model.") for k in state_dict):
        state_dict = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }

    return state_dict


def main(
    wav_path: str,
    output_path: str,
    model_path: str,
    model_complexity: int = 48,
    sequence_length: int = 327680,
    device: str = "cpu",
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
):
    device = torch.device(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = fix_state_dict(state_dict)
    model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity)
    model.load_state_dict(state_dict)
    model.eval()

    audio = load_audio(wav_path)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=WINDOW_LENGTH,
        win_length=WINDOW_LENGTH,
        hop_length=HOP_LENGTH,
        f_min=MEL_FMIN,
        f_max=MEL_FMAX,
        norm="slaney",
        power=1.0,
    ).to(device)

    n_steps = (len(audio) - 1) // HOP_LENGTH + 1

    onset_pred_all = torch.zeros((n_steps, MAX_MIDI - MIN_MIDI + 1))
    offset_pred_all = torch.zeros((n_steps, MAX_MIDI - MIN_MIDI + 1))
    frame_pred_all = torch.zeros((n_steps, MAX_MIDI - MIN_MIDI + 1))
    velocity_pred_all = torch.zeros((n_steps, MAX_MIDI - MIN_MIDI + 1))

    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(audio), sequence_length)):
            step = i // HOP_LENGTH
            x = audio[i : i + sequence_length].to(device)
            mel = mel_transform(x.reshape(-1, x.shape[-1])[:, :-1])
            mel = torch.log(torch.clamp(mel, min=1e-5))
            mel = mel.transpose(-1, -2)
            onset_pred, offset_pred, _, frame_pred, velocity_pred = model(mel)

            onset_pred = (
                onset_pred.reshape((onset_pred.shape[1], onset_pred.shape[2]))
                .detach()
                .cpu()
            )
            offset_pred = (
                offset_pred.reshape((offset_pred.shape[1], offset_pred.shape[2]))
                .detach()
                .cpu()
            )
            frame_pred = (
                frame_pred.reshape((frame_pred.shape[1], frame_pred.shape[2]))
                .detach()
                .cpu()
            )
            velocity_pred = (
                velocity_pred.reshape((velocity_pred.shape[1], velocity_pred.shape[2]))
                .detach()
                .cpu()
            )

            onset_pred_all[step : step + onset_pred.shape[0]] = onset_pred
            offset_pred_all[step : step + offset_pred.shape[0]] = offset_pred
            frame_pred_all[step : step + frame_pred.shape[0]] = frame_pred
            velocity_pred_all[step : step + velocity_pred.shape[0]] = velocity_pred

    scaling = HOP_LENGTH / SAMPLE_RATE
    notes = extract_notes(
        onset_pred_all,
        frame_pred_all,
        velocity_pred_all,
        scaling,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )

    if len(notes) == 0:
        print("No notes found.")
        return

    save_pianoroll(
        output_path + ".png",
        onset_pred_all,
        frame_pred_all,
        onset_threshold,
        frame_threshold,
    )

    midi = create_midi(notes)
    midi.write(output_path)

    pass


if __name__ == "__main__":
    fire.Fire(main)
