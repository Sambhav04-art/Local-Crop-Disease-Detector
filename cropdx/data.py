import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .config import path_of
from .perturb import RandomBackground, RandomJPEG, list_images
from .preprocess import MEAN, STD, load_rgb_from_path, preprocess_pil


class EvalTransform:
    """Exactly the API preprocessing (resize -> normalise)."""
    def __init__(self, size=224):
        self.size = size

    def __call__(self, img):
        return torch.from_numpy(preprocess_pil(img, self.size))


def build_train_transform(cfg):
    a, s = cfg["augment"], cfg["image"]["size"]
    bg_dir = path_of(cfg, "backgrounds_dir", "data/backgrounds")
    n_bg = len(list_images(bg_dir))
    print(f"Real background photos found in {bg_dir}: {n_bg}" + ("" if n_bg else "  (none -> synthetic backgrounds only; see README 'Make it work on real photos')"))
    return T.Compose([
        RandomBackground(a["background_prob"], bg_dir, a.get("real_background_prob", 0.7), a.get("leaf_shrink_min", 0.55)),
        RandomJPEG(a["jpeg_prob"]),
        T.RandomResizedCrop(s, scale=(0.6, 1.0), ratio=(0.85, 1.15)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(30),
        T.ColorJitter(a["brightness"], a["contrast"], a["saturation"], a["hue"]),
        T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=a["blur_prob"]),
        T.ToTensor(), T.Normalize(MEAN, STD),
        T.RandomErasing(p=a["erasing_prob"], scale=(0.02, 0.15)),
    ])


class ManifestDataset(Dataset):
    def __init__(self, df, transform):
        self.paths, self.y, self.transform = df.path.tolist(), df.class_idx.to_numpy(), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_rgb_from_path(self.paths[i])
        return self.transform(img), int(self.y[i])


def subsample_idx(df, n_per_class, seed=0):
    """Integer positions of at most n images per class (stratified, reproducible)."""
    rng, y, idx = np.random.default_rng(seed), df.class_idx.to_numpy(), []
    for c in np.unique(y):
        pos = np.nonzero(y == c)[0]
        idx.extend(rng.choice(pos, min(n_per_class, len(pos)), replace=False))
    return np.sort(np.array(idx))
