import torchaudio

from modules.constants import SAMPLE_RATE


def load_audio(path: str) :
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.mean(0)
    return wav