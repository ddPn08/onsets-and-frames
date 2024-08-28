import torch
from PIL import Image


def save_pianoroll(
    path: str,
    onsets: torch.Tensor,
    frames: torch.Tensor,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
    zoom: float = 4,
):
    """
    Saves a piano roll diagram

    Parameters
    ----------
    path: str
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    zoom: int
    """
    onsets = (1 - (onsets.t() > onset_threshold).to(torch.uint8)).cpu()
    frames = (1 - (frames.t() > frame_threshold).to(torch.uint8)).cpu()
    both = 1 - (1 - onsets) * (1 - frames)
    image = torch.stack([onsets, frames, both], dim=2).flip(0).mul(255).numpy()
    image = Image.fromarray(image, "RGB")
    image = image.resize((image.size[0], image.size[1] * zoom))
    image.save(path)
