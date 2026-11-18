"""EuroSAT class names, prompt templates, splits and few-shot sampling.

Label order follows torchvision's EuroSAT (ImageFolder, alphabetical folder names).
English class names are the ones CoOp uses for EuroSAT (CoOp repo, datasets/eurosat.py,
NEW_CNAMES). Turkish names are our own translations (see README: verify them).
"""
import json

import numpy as np
from sklearn.model_selection import train_test_split

FOLDERS = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

CLASSNAMES = {
    "en": [
        "Annual Crop Land", "Forest", "Herbaceous Vegetation Land", "Highway or Road",
        "Industrial Buildings", "Pasture Land", "Permanent Crop Land",
        "Residential Buildings", "River", "Sea or Lake",
    ],
    "tr": [
        "Tek Yıllık Tarım Arazisi", "Orman", "Otsu Bitki Örtüsü Alanı", "Otoyol veya Yol",
        "Sanayi Binaları", "Mera Arazisi", "Çok Yıllık Tarım Arazisi",
        "Konut Binaları", "Nehir", "Deniz veya Göl",
    ],
}

# Zero-shot templates. English = CoOp's hand-crafted EuroSAT template.
# Turkish uses a colon so the class name needs no case suffix (Turkish is agglutinative).
TEMPLATES = {
    "en": "a centered satellite photo of {}.",
    "tr": "ortalanmış bir uydu fotoğrafı: {}.",
}

LANGS = tuple(CLASSNAMES)


def make_splits(labels, rel_paths=None, split_json=None, seed=0, fractions=(0.5, 0.2, 0.3)):
    """Return {'train','val','test'} index arrays.

    With split_json (CoOp's split_zhou_EuroSAT.json) the official split is used.
    Otherwise a stratified split with the same proportions as CoOp's
    (13,500 / 5,400 / 8,100 of 27,000 images) is drawn with a fixed seed.
    """
    labels = np.asarray(labels)
    if split_json:
        with open(split_json) as f:
            sp = json.load(f)
        index = {p.replace("\\", "/"): i for i, p in enumerate(rel_paths)}
        out = {}
        for k in ("train", "val", "test"):
            out[k] = np.array(sorted(index[item[0].replace("\\", "/")] for item in sp[k]))
        return out
    idx = np.arange(len(labels))
    n = len(labels)
    n_tr = int(round(fractions[0] * n))
    n_va = int(round(fractions[1] * n))
    n_te = n - n_tr - n_va
    tr, rest = train_test_split(idx, train_size=n_tr, stratify=labels, random_state=seed)
    va, te = train_test_split(rest, train_size=n_va, test_size=n_te,
                              stratify=labels[rest], random_state=seed)
    return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}


def sample_few_shot(pool_idx, labels, k, seed):
    """Pick k examples per class from pool_idx (deterministic for a given seed)."""
    labels = np.asarray(labels)
    pool_idx = np.asarray(pool_idx)
    rng = np.random.RandomState(seed)
    out = []
    for c in np.unique(labels[pool_idx]):
        cand = pool_idx[labels[pool_idx] == c]
        out.extend(rng.choice(cand, size=min(k, len(cand)), replace=False).tolist())
    return np.array(sorted(out))
