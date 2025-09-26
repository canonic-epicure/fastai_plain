import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms import v2


class DatasetFix(Dataset):
    def __getitem__(self, idx):
        return {}


class DatasetWithLabel(DatasetFix):
    def __init__(self, *args, labels: pd.DataFrame=None, **kwargs):
        self.labels = labels
        super().__init__(*args, **kwargs)

    def get_label(self, idx):
        return torch.tensor(self.labels.iloc[idx], dtype=torch.long)

    def __getitem__(self, idx):
        sup = super().__getitem__(idx)

        sup['labels'] = self.get_label(idx)

        return sup


class ImageDataset(DatasetFix):
    def __init__(self, *args, data: pd.Series, processor, aug=None, **kwargs):
        self.data = data

        self.processor = processor
        self.aug = aug

        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.data)

    def get_image(self, idx):
        image_id = self.data.index[idx]

        image = Image.open(self.data.iloc[idx]).convert("RGB")

        image = self.aug(image) if self.aug else image
        image = self.processor(image)

        return image, image_id

    def __getitem__(self, idx):
        sup = super().__getitem__(idx)

        image, image_id = self.get_image(idx)

        sup['inputs'] = image
        sup['ids'] = image_id

        return sup


class ImageDatasetWithLabel(DatasetWithLabel, ImageDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        return super().__getitem__(idx)


def resize_to_square_with_reflect(img: Image.Image, size: int, as_tensor=True) -> Image.Image:
    """
    Масштабирует изображение, сохраняя аспект, так чтобы max(H, W)=size,
    затем дополняет до size×size отражающим паддингом (reflect).
    Возвращает PIL.Image.
    """
    # → Tensor [C,H,W], float32
    t = F.pil_to_tensor(img).float() / 255.0 if isinstance(img, Image.Image) else img
    if t.ndim == 2:  # grayscale H×W → 1×H×W
        t = t.unsqueeze(0)

    _, h, w = t.shape
    if h == 0 or w == 0:
        raise ValueError("Empty image")

    scale = size / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # resize с сохранением аспекта
    t = F.resize(t, [new_h, new_w], antialias=True, interpolation = InterpolationMode.BICUBIC)

    # паддинги (left, top, right, bottom) так, чтобы центрировать картинку
    pad_top    = (size - new_h) // 2
    pad_bottom = size - new_h - pad_top
    pad_left   = (size - new_w) // 2
    pad_right  = size - new_w - pad_left

    # reflect-паддинг
    # Важно: в PyTorch для 'reflect' величина паддинга на сторону должна быть < соответствующего размера (new_h/new_w).
    # Если получите ошибку "Padding size should be less than the corresponding input dimension",
    # используйте вариант B ниже (через NumPy), он без этого ограничения.
    t = F.pad(t, [pad_left, pad_top, pad_right, pad_bottom], padding_mode='reflect')

    t = t.clamp(0, 1)

    # обратно в PIL
    return t if as_tensor else F.to_pil_image(t)
