"""
Experiment runner for the event + dense cross-modal autoencoder on DSERT-RoLL.
Sits next to event_dense_ae.py (the model) and dsert_stream.py (data access).

    python run_experiments.py cache --peek            # look at label.pkl + sensor formats
    python run_experiments.py cache                   # download + cache (resumable)
    python run_experiments.py labels                  # rebuild 2D boxes offline if needed
    python run_experiments.py inspect                 # check shapes, labels, split, alignment
    python run_experiments.py overlap                 # heatmap-box overlap (no training)
    python run_experiments.py pretrain                # every entry in EXPERIMENTS
    python run_experiments.py probe --train-fusion    # detector at 10/25/50/100 % labels
    python run_experiments.py plot                    # label-efficiency curve + 2x2 table

Layout of this file:
    MODEL        which model is pretrained and probed  <- swap models here
    EXPERIMENTS  the runs and label fractions          <- add/remove runs here
    DETECTOR     the FCOS head used by `probe`
    data, caching, metrics and the commands follow.
See README.md for every option.
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import re
import time
from collections import OrderedDict, defaultdict

import numpy as np

try:                                   # torch-free parts (parsing, AP) stay importable
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:                    # pragma: no cover
    torch = None
    Dataset = object

NAME_RE = re.compile(
    r"^(?P<cond>.+?)__(?P<seq>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})__"
    r"(?P<mod>rgb|thermal|event)_(?P<side>[lr])__(?P<hash>[0-9a-fA-F]+)\.npy$")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# =============================================================================
# MODEL  -- the only place that names a specific model. Built-in choices, picked
# with --model:  crossmae  (event_dense_ae.py, ResNet-18 encoders)
#                videomae  (videomae_ae.py, pretrained VideoMAE encoders)
# To add another model, add a branch to build_model() (and, if it trains
# differently, to train_step()); nothing else in this file needs to change.
#
# The model object returned by build_model() must provide:
#   forward(ev, dn, hide)  -> reconstruction of the hidden modality ("event"/"dense")
#   encode(ev, dn)         -> fused features (B, out_channels, H/16, W/16)
#   out_channels           -> int, channels of encode()'s output
#   fusion_modules()       -> list of sub-modules `probe --train-fusion` trains
#                             (return [] if the model has no separate fusion part)
# ev is the event voxel grid (B, event_bins, H, W); dn the dense frame (B, dense_ch, H, W).
# =============================================================================
MODELS = ["crossmae", "videomae"]


def build_model(name, dense_ch, event_bins, imagenet=True, size=(256, 384)):
    """A fresh model that has not been pretrained by this project.
    imagenet=True starts its encoders from their public pretrained weights
    (ImageNet for crossmae, Kinetics VideoMAE for videomae)."""
    if name == "crossmae":
        from event_dense_ae import CrossModalAE
        return CrossModalAE(dense_ch=dense_ch, event_bins=event_bins, imagenet=imagenet)
    if name == "videomae":
        from videomae_ae import VideoMAEAE
        return VideoMAEAE(dense_ch=dense_ch, event_bins=event_bins, imagenet=imagenet, size=size)
    raise ValueError(f"unknown --model {name!r}; add it to build_model()")


def train_step(model, opt, ev, dn, cfg):
    """One pretraining step; cfg is one EXPERIMENTS entry. Returns the loss."""
    from event_dense_ae import pretrain_step
    return pretrain_step(model, opt, ev, dn, **cfg)


def weight_map(ev, mode):
    """(B,1,H,W) loss weights from events; used by `overlap` (and by train_step)."""
    from event_dense_ae import event_weight_map
    return event_weight_map(ev, mode)


WEIGHT_MODES = ["none", "count", "floor", "recency", "ego"]     # compared by `overlap`


# =============================================================================
# EXPERIMENTS  -- every pretraining run. The dict values are passed to
# train_step() as cfg. None = no pretraining (a baseline built in `probe`).
# `probe` also adds the "a_imagenet" baseline (ImageNet weights, no pretraining).
# =============================================================================
EXPERIMENTS = {
    # professor's three-way comparison: (a) random init, (b) plain, (c) weighted
    "a_random_init": None,
    "b_plain":       dict(direction="random", weight_mode="none"),
    "c_count":       dict(direction="random", weight_mode="count"),
    "c_floor":       dict(direction="random", weight_mode="floor"),
    "c_recency":     dict(direction="random", weight_mode="recency"),
    "c_ego":         dict(direction="random", weight_mode="ego"),
    # 2x2: mask direction x weighting
    "e2d_plain":     dict(direction="e2d", weight_mode="none"),
    "e2d_weighted":  dict(direction="e2d", weight_mode="ego"),
    "d2e_plain":     dict(direction="d2e", weight_mode="none"),
    "d2e_weighted":  dict(direction="d2e", weight_mode="ego"),
}
LABEL_FRACTIONS = [0.10, 0.25, 0.50, 1.00]      # override with probe --fractions


# =============================================================================
# DETECTOR  -- FCOS head on top of model.encode(); used only by `probe`.
# A fresh detector is built, trained and discarded for every probe run.
# =============================================================================
_Module = torch.nn.Module if torch else object


class FusedBackbone(_Module):
    """Wraps the model so torchvision FCOS can use it as a backbone.
    FCOS takes one image tensor, so event and dense channels arrive stacked:
    x = cat([event_voxel_grid, dense_frame], dim=1).
    freeze=True:        model weights fixed (no gradients, BatchNorm stats fixed)
    train_fusion=True:  ...except model.fusion_modules(), which train with the head"""

    def __init__(self, model, event_bins, freeze=True, train_fusion=False):
        super().__init__()
        nn = torch.nn
        self.model, self.event_bins = model, event_bins
        self.freeze, self.train_fusion = freeze, train_fusion
        C = self.out_channels = model.out_channels
        # trainable extra levels. p3 (stride 8) exists so that small objects
        # (pedestrians, distant cars) contain at least one prediction point.
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 nn.Conv2d(C, C, 3, padding=1))
        self.down5 = nn.Conv2d(C, C, 3, 2, 1)
        self.down6 = nn.Conv2d(C, C, 3, 2, 1)
        if freeze:
            model.requires_grad_(False)
            if train_fusion:
                for m in model.fusion_modules():
                    m.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:                         # keep BatchNorm statistics frozen too
            self.model.eval()
            if self.train_fusion and mode:
                for m in self.model.fusion_modules():
                    m.train()
        return self

    def forward(self, x):
        ev, dn = x[:, :self.event_bins], x[:, self.event_bins:]
        p4 = self.model.encode(ev, dn)                                # stride 16
        p3 = self.up3(p4)                                             # stride 8
        p5 = self.down5(F.relu(p4))                                   # stride 32
        p6 = self.down6(F.relu(p5))                                   # stride 64
        return OrderedDict([("p3", p3), ("p4", p4), ("p5", p5), ("p6", p6)])


def build_detector(model, num_classes, dense_ch, event_bins, size, freeze=True,
                   train_fusion=False, score_thresh=0.05, head_convs=4):
    """torchvision FCOS on FusedBackbone(model).
    num_classes includes background (Bike, Pedestrian, Vehicle -> 4; labels 1..3).
    Anchor size = stride on every level (torchvision's FCOS convention): objects
    up to ~64 px go to p3, 64-128 to p4, 128-256 to p5, larger to p6.
    score_thresh: torchvision's default 0.2 hides most detections of a weakly
      trained head (mAP reads ~0); 0.05 is the usual evaluation setting.
    head_convs: 3x3 conv layers in each head branch. 4 = standard FCOS,
      1 = shallow probe, 0 = single conv per output (closest to a linear probe)."""
    from torchvision.models.detection import FCOS
    from torchvision.models.detection.fcos import FCOSHead
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    c = event_bins + dense_ch
    backbone = FusedBackbone(model, event_bins, freeze, train_fusion)
    anchors = AnchorGenerator(sizes=((8,), (16,), (32,), (64,)), aspect_ratios=((1.0,),) * 4)
    head = FCOSHead(backbone.out_channels, 1, num_classes, num_convs=head_convs)
    return FCOS(backbone, num_classes=num_classes, anchor_generator=anchors, head=head,
                image_mean=[0.0] * c, image_std=[1.0] * c, min_size=min(size),
                max_size=max(size), score_thresh=score_thresh, detections_per_img=100)


# ============================================================ cache discovery
def parse_cache_name(fname):
    m = NAME_RE.match(os.path.basename(fname))
    return m.groupdict() if m else None


def find_groups(cache_dir, dense="thermal", side="l"):
    """-> list of dicts {cond, seq, hash, event: path, dense: path}, only for
    (sequence, hash) pairs that have BOTH the event and dense file."""
    files = defaultdict(dict)
    meta = {}
    for f in sorted(glob.glob(os.path.join(cache_dir, "*.npy"))):
        info = parse_cache_name(f)
        if not info or info["side"] != side:
            continue
        key = (info["seq"], info["hash"])
        files[key][info["mod"]] = f
        meta[key] = info["cond"]
    groups = []
    for (seq, h), mods in sorted(files.items()):
        if "event" in mods and dense in mods:
            groups.append(dict(cond=meta[(seq, h)], seq=seq, hash=h,
                               event=mods["event"], dense=mods[dense]))
    return groups


# ================================================================ array layout
def detect_layout(shape):
    """Guess how frames are stored. Returns one of NHW, NHWC, NCHW, HW, HWC."""
    if len(shape) == 2:
        return "HW"
    if len(shape) == 3:
        if shape[-1] <= 16 and shape[0] >= 32 and shape[1] >= 32:
            return "HWC"            # a single frame with channels last
        return "NHW"
    if len(shape) == 4:
        return "NHWC" if (shape[-1] <= 16 and shape[1] > 16) else "NCHW"
    if len(shape) == 5:             # clips of frames, e.g. (4, 16, 112, 224, 3)
        return "KTHWC" if shape[-1] <= 16 else "KTCHW"
    raise ValueError(f"unsupported array shape {shape}")


def num_frames(shape, layout):
    if layout in ("HW", "HWC"):
        return 1
    if layout in ("KTHWC", "KTCHW"):
        return shape[0] * shape[1]          # clips x frames per clip, clip-major
    return shape[0]


def get_frame(arr, i, layout):
    """-> float32 numpy (C,H,W) for frame i."""
    if layout in ("KTHWC", "KTCHW"):
        x = arr[i // arr.shape[1], i % arr.shape[1]]
        return np.asarray(np.moveaxis(x, -1, 0) if layout == "KTHWC" else x, dtype=np.float32)
    if layout == "HW":
        x = arr[None]
    elif layout == "HWC":
        x = np.moveaxis(arr, -1, 0)
    elif layout == "NHW":
        x = arr[i][None]
    elif layout == "NHWC":
        x = np.moveaxis(arr[i], -1, 0)
    else:
        x = arr[i]
    return np.asarray(x, dtype=np.float32)


def normalize(x, mod):
    if mod == "rgb":
        if x.max() > 1.5:
            x = x / 255.0
        if x.shape[0] == 3:
            x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        return x
    if mod == "thermal":
        return (x - x.mean()) / (x.std() + 1e-6)
    return x / (x.std() + 1e-6)          # events: keep sign and sparsity


# ================================================================== labels
def load_coco_labels(dataset_dir, sequences):
    """-> (labels, class_names) where labels[seq] = list over frames (sorted by
    file_name) of dict(boxes=np (K,4) xyxy in ORIGINAL pixels, labels=np (K,),
    size=(h, w)). Class ids are remapped to 1..C (0 = background for FCOS)."""
    jsons = glob.glob(os.path.join(dataset_dir, "**", "*.json"), recursive=True)
    cat_names, per_seq = {}, defaultdict(dict)
    for jp in jsons:
        try:
            with open(jp) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if not (isinstance(d, dict) and "images" in d and "annotations" in d):
            continue
        for c in d.get("categories", []):
            cat_names.setdefault(c["id"], c["name"])
        seq_from_path = next((s for s in sequences if s in jp), None)
        anns = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
        for im in d["images"]:
            seq = seq_from_path or next((s for s in sequences if s in im["file_name"]), None)
            if seq is None:
                continue
            per_seq[seq][im["file_name"]] = (im, anns.get(im["id"], []))
    ids = sorted(cat_names)
    remap = {cid: k + 1 for k, cid in enumerate(ids)}
    labels = {}
    for seq, entries in per_seq.items():
        frames = []
        for fname in sorted(entries):
            im, al = entries[fname]
            boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                               a["bbox"][1] + a["bbox"][3]] for a in al], np.float32).reshape(-1, 4)
            cls = np.array([remap[a["category_id"]] for a in al], np.int64)
            frames.append(dict(boxes=boxes, labels=cls, size=(im["height"], im["width"])))
        labels[seq] = frames
    return labels, [cat_names[c] for c in ids]


# ============================================================= splits/fractions
def split_sequences(groups, val_frac=0.2, seed=0, cache_dir=None):
    """Official DSERT-RoLL split if `cache` wrote <cache>/splits.json,
    otherwise validation = ~val_frac of whole sequences, spread across weather."""
    sp = os.path.join(cache_dir, "splits.json") if cache_dir else None
    if sp and os.path.exists(sp):
        with open(sp) as fh:
            off = json.load(fh)
        have = {g["seq"] for g in groups}
        pick = lambda name: sorted({s.split("/")[-1] for s in off[name]} & have)
        if pick("train") and pick("val"):
            return pick("train"), pick("val")
    by_cond = defaultdict(set)
    for g in groups:
        by_cond[g["cond"]].add(g["seq"])
    rng = random.Random(seed)
    lists = []
    for cond in sorted(by_cond):
        s = sorted(by_cond[cond])
        rng.shuffle(s)
        lists.append(s)
    if not lists:
        return [], []
    interleaved = [s[i] for i in range(max(map(len, lists))) for s in lists if i < len(s)]
    n_val = max(1, math.ceil(val_frac * len(interleaved)))   # spread across conditions
    val = set(interleaved[:n_val])
    train = sorted({g["seq"] for g in groups} - val)
    return train, sorted(val)


def label_subset(index, fraction, block=50, seed=0):
    """Nested subsets of contiguous frame blocks (neighbouring frames are near
    duplicates, so sample blocks, not frames). index: list of (gid, frame)."""
    if fraction >= 1.0:
        return list(index)
    blocks = sorted({(g, f // block) for g, f in index})
    random.Random(seed).shuffle(blocks)              # same order for every fraction
    keep = set(blocks[:max(1, math.ceil(fraction * len(blocks)))])
    return [(g, f) for g, f in index if (g, f // block) in keep]


# =================================================================== dataset
class CachedPairs(Dataset):
    """Yields (event (Ce,H,W), dense (Cd,H,W), target or None)."""

    def __init__(self, groups, seqs, dense="thermal", size=(384, 480), labels=None,
                 require_labels=False):
        self.groups = [g for g in groups if g["seq"] in set(seqs)]
        self.dense, self.size, self.labels = dense, tuple(size), labels
        self._arrays = {}
        self.index = []
        for gi, g in enumerate(self.groups):
            n = min(self._n(g["event"]), self._n(g["dense"]))
            lab = labels.get(g["seq"]) if labels else None
            if lab is not None and len(lab) != n:
                print(f"  [warn] {g['seq']} {g['hash']}: {n} cached frames vs "
                      f"{len(lab)} labelled images -> using first {min(n, len(lab))}")
                n = min(n, len(lab))
            if require_labels and lab is None:
                continue
            self.index += [(gi, f) for f in range(n)]

    def _load(self, path):
        if path not in self._arrays:                 # lazy, per worker process
            arr = np.load(path, mmap_mode="r")
            self._arrays[path] = (arr, detect_layout(arr.shape))
        return self._arrays[path]

    def _n(self, path):
        arr, lay = self._load(path)
        return num_frames(arr.shape, lay)

    def _frame(self, path, f, mod):
        arr, lay = self._load(path)
        x = torch.from_numpy(normalize(get_frame(arr, f, lay), mod))[None]
        return F.interpolate(x, size=self.size, mode="bilinear", align_corners=False,
                             antialias=True)[0]

    def __len__(self):
        return len(self.index)

    def get(self, gi, f):
        g = self.groups[gi]
        ev = self._frame(g["event"], f, "event")
        dn = self._frame(g["dense"], f, self.dense)
        tgt = None
        if self.labels and g["seq"] in self.labels:
            lab = self.labels[g["seq"]][f]
            h0, w0 = lab["size"]
            sy, sx = self.size[0] / h0, self.size[1] / w0
            b = torch.from_numpy(lab["boxes"].copy()) * torch.tensor([sx, sy, sx, sy])
            H, W = self.size
            b[:, 0::2] = b[:, 0::2].clamp(0, W)
            b[:, 1::2] = b[:, 1::2].clamp(0, H)
            ok = ((b[:, 2] - b[:, 0]) >= 1) & ((b[:, 3] - b[:, 1]) >= 1)   # FCOS rejects
            lbl = torch.from_numpy(lab["labels"].copy())                    # degenerate boxes
            tgt = dict(boxes=b[ok].float(), labels=lbl[ok])
        return ev, dn, tgt

    def __getitem__(self, i):
        return self.get(*self.index[i])


class Subset(Dataset):
    def __init__(self, base, index):
        self.base, self.index = base, index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        return self.base.get(*self.index[i])


def collate(batch):
    ev = torch.stack([b[0] for b in batch])
    dn = torch.stack([b[1] for b in batch])
    return ev, dn, [b[2] for b in batch]


def _worker_init(_):
    # Data-loading workers are forked copies of the main process. If the main
    # process already started its OpenMP thread pool (CPU interpolation in
    # CachedPairs), a forked copy can deadlock the first time it uses it, which
    # leaves the whole job idle. One thread per worker avoids that pool.
    torch.set_num_threads(1)


def train_loader(ds, args):
    """Shuffled training loader. timeout: if no batch arrives for 10 minutes
    (a stuck worker), raise an error instead of sitting idle on the GPU."""
    w = args.workers
    return DataLoader(ds, args.batch, shuffle=True, num_workers=w, collate_fn=collate,
                      drop_last=True, persistent_workers=w > 0,
                      worker_init_fn=_worker_init if w > 0 else None,
                      timeout=600 if w > 0 else 0)


def infinite(loader):
    while True:
        for b in loader:
            yield b


# =================================================================== metrics
def iou_matrix(a, b):
    """numpy (N,4) x (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def average_precision(dets, gts, cls, iou_thr=0.5):
    """VOC-style all-point AP for one class.
    dets: list over images of (boxes, scores, labels); gts: list of (boxes, labels)."""
    records, n_gt, gt_c = [], 0, []
    for img, (gb, gl) in enumerate(gts):
        g = gb[gl == cls]
        gt_c.append(g)
        n_gt += len(g)
    if n_gt == 0:
        return float("nan")
    for img, (db, ds, dl) in enumerate(dets):
        m = dl == cls
        records += [(s, img, b) for s, b in zip(ds[m], db[m])]
    records.sort(key=lambda r: -r[0])
    used = [np.zeros(len(g), bool) for g in gt_c]
    tp = np.zeros(len(records))
    for k, (_, img, box) in enumerate(records):
        g = gt_c[img]
        if len(g):
            ious = iou_matrix(box[None], g)[0]
            j = int(ious.argmax())
            if ious[j] >= iou_thr and not used[img][j]:
                used[img][j] = True
                tp[k] = 1
    if len(records) == 0:
        return 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def heatmap_box_overlap(w, boxes, h, wd):
    """w: (H,W) weight map; boxes (K,4) in the same pixels.
    mass  = share of heatmap mass inside the boxes
    area  = share of image area covered by boxes
    lift  = mass / area   (1.0 = no better than uniform)"""
    mask = np.zeros((h, wd), bool)
    for x0, y0, x1, y1 in boxes:
        mask[max(0, int(y0)):min(h, int(math.ceil(y1))), max(0, int(x0)):min(wd, int(math.ceil(x1)))] = True
    area = mask.mean()
    if area == 0:
        return None
    mass = float(w[mask].sum() / (w.sum() + 1e-9))
    return mass, float(area), mass / float(area)


# ================================================================== helpers
def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_rows(path, rows):
    """Append rows to a CSV. If the new rows have columns the file doesn't
    (e.g. probe settings added later), the file is rewritten with the union
    of columns so old and new rows stay aligned; missing cells are left empty."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(rows[0].keys())
    if os.path.exists(path):
        with open(path, newline="") as fh:
            rd = csv.DictReader(fh)
            old_keys, old_rows = list(rd.fieldnames or []), list(rd)
        if all(k in old_keys for k in keys):
            with open(path, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=old_keys, restval="").writerows(rows)
            return
        fields = old_keys + [k for k in keys if k not in old_keys]
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields, restval="")
            wr.writeheader()
            wr.writerows(old_rows + list(rows))
        return
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def channels(ds):
    ev, dn, _ = ds[0]
    return ev.shape[0], dn.shape[0]


def load_labels(args, sequences):
    if glob.glob(os.path.join(args.cache, "labels", "*.json")):
        return load_cached_labels(args.cache, sequences)
    return load_coco_labels(args.dataset, sequences)


def no_cache_exit(args):
    path = os.path.abspath(args.cache)
    n = len(glob.glob(os.path.join(path, "*.npy")))
    raise SystemExit(
        f"\nNo complete event+{args.dense} cache found in:\n  {path}\n"
        f"({'folder does not exist' if not os.path.isdir(path) else f'{n} .npy files there'})\n"
        "Copy dsert_frames_cache/ next to run_experiments.py (or pass --cache /path/to/it),\n"
        "or build it with `python run_experiments.py cache`.")


def setup(args, need_labels=False):
    groups = find_groups(args.cache, args.dense)
    if not groups:
        no_cache_exit(args)
    train, val = split_sequences(groups, seed=args.seed, cache_dir=args.cache)
    labels, names = (load_labels(args, sorted({g['seq'] for g in groups}))
                     if need_labels else (None, []))
    if need_labels and not labels:
        raise SystemExit("No 2D labels found. Run `cache --peek`, adjust extract_boxes_2d(), "
                         "then `labels`.")
    return groups, train, val, labels, names


# ================================================================ HF caching
# Builds the cache this runner reads, straight from Hugging Face, using
# dsert_stream.py (keep it next to this file). Differences from the notebook's
# VideoMAE clip cache:
#   * individual frames every --frame-step frames (not 16-frame clips)
#   * events -> 5-bin voxel grid of RAW signed counts (not a saturated gray image)
#   * thermal -> 1 channel float, rgb -> 3 channel uint8
#   * the sequence's label.pkl is saved too, so 2D boxes can be rebuilt offline
#   * official train/val split is recorded in <cache>/splits.json
SENSOR_KEY = {"event": "event_l", "thermal": "thermal_l", "rgb": "rgb_l"}


def describe(o, depth=0, maxd=4, key="", maxkeys=30):
    """Print the structure of nested dict/list/array data (for --peek)."""
    pad = "  " * depth
    if isinstance(o, dict):
        print(f"{pad}{key}dict[{len(o)}]")
        if depth < maxd:
            for k in list(o)[:maxkeys]:
                describe(o[k], depth + 1, maxd, f"{k!r}: ", maxkeys)
    elif isinstance(o, (list, tuple)):
        print(f"{pad}{key}{type(o).__name__}[{len(o)}]")
        if o and depth < maxd:
            describe(o[0], depth + 1, maxd, "[0]: ", maxkeys)
    elif isinstance(o, np.ndarray):
        extra = ""
        if o.size and o.dtype.kind in "iuf":
            extra = f" min={o.min():.5g} max={o.max():.5g} first={o.ravel()[:6].tolist()}"
        elif o.size and o.dtype.kind in "USO":
            extra = f" first={o.ravel()[:6].tolist()}"
        print(f"{pad}{key}ndarray{o.shape} {o.dtype}{extra}")
    else:
        print(f"{pad}{key}{type(o).__name__}: {repr(o)[:100]}")


def parse_events(d):
    """Event .npz contents -> (x, y, t, p) float64 arrays, p in {-1, +1}.
    Handles named x/y/t/p arrays or one N x 4 (or N x 3) array whose columns are
    identified by content: polarity has <= 2 values, time has the largest
    values, and x has a larger range than y (sensor is wider than tall)."""
    if isinstance(d, dict):
        keys = {k.lower(): k for k in d}
        g = lambda *ns: next((np.asarray(d[keys[n]]) for n in ns if n in keys), None)
        x, y = g("x", "xs"), g("y", "ys")
        if x is not None and y is not None:
            t = g("t", "ts", "timestamp", "timestamps")
            p = g("p", "ps", "pol", "polarity")
            t = np.arange(len(x), dtype=np.float64) if t is None else t.astype(np.float64)
            p = np.ones(len(x)) if p is None else np.where(p > 0, 1.0, -1.0)
            return x.astype(np.float64), y.astype(np.float64), t, p
        a = max((np.asarray(v) for v in d.values()), key=lambda v: v.size)
    else:
        a = np.asarray(d)
    if a.ndim != 2 or a.shape[1] not in (3, 4):
        raise ValueError(f"don't know how to read events of shape {a.shape}; see --peek")
    cols = [a[:, i].astype(np.float64) for i in range(a.shape[1])]
    sample = slice(0, min(len(a), 50000))
    pi = next((i for i, c in enumerate(cols) if len(np.unique(c[sample])) <= 2), None)
    rest = [i for i in range(len(cols)) if i != pi]
    ti = None
    if len(rest) == 3:
        ti = max(rest, key=lambda i: cols[i].max())
        rest.remove(ti)
    xi, yi = sorted(rest, key=lambda i: cols[i].max(), reverse=True)
    t = cols[ti] if ti is not None else np.arange(len(a), dtype=np.float64)
    p = np.where(cols[pi] > 0, 1.0, -1.0) if pi is not None else np.ones(len(a))
    return cols[xi], cols[yi], t, p


def voxel_grid_fast(x, y, t, p, sensor_hw, out_hw, bins=5, H_mat=None):
    """Events -> (bins, H, W) float32 of signed counts, binned straight into the
    output resolution (equivalent to sum-pooling, so counts are preserved).
    Optional H_mat (3x3) warps sensor pixels into another camera's plane."""
    if H_mat is not None:
        q = H_mat @ np.stack([x, y, np.ones_like(x)])
        x, y = q[0] / q[2], q[1] / q[2]
    Hs, Ws = sensor_hw
    Ho, Wo = out_hw
    xi = np.floor(x * Wo / Ws).astype(np.int64)
    yi = np.floor(y * Ho / Hs).astype(np.int64)
    ok = (xi >= 0) & (xi < Wo) & (yi >= 0) & (yi < Ho)
    xi, yi, t, p = xi[ok], yi[ok], t[ok], p[ok]
    if len(t) == 0:
        return np.zeros((bins, Ho, Wo), np.float32)
    tn = (t - t.min()) / max(t.max() - t.min(), 1e-9) * (bins - 1)
    t0 = np.floor(tn).astype(np.int64)
    grid = np.zeros(bins * Ho * Wo, np.float64)
    for b, w in ((t0, 1 - (tn - t0)), (t0 + 1, tn - t0)):
        m = b < bins
        grid += np.bincount(b[m] * Ho * Wo + yi[m] * Wo + xi[m], weights=p[m] * w[m],
                            minlength=bins * Ho * Wo)
    return grid.reshape(bins, Ho, Wo).astype(np.float32)


def _resize(a, hw, mode="bilinear"):
    from PIL import Image
    im = Image.fromarray(np.ascontiguousarray(a, dtype=np.float32))
    return np.asarray(im.resize((hw[1], hw[0]), Image.BILINEAR), dtype=np.float32)


def _boxes_from_array(a, box_format="auto"):
    """One camera's gt_boxes_* entry -> (K,4) xyxy, or None if the layout is unknown.
    Accepts (K,4) boxes, (K,N,2|3) projected corners, (K,2N) flattened corners,
    or a list/object array of per-object corner arrays."""
    a = np.asarray(a)
    if a.dtype == object:
        try:
            rows = [np.asarray(o, np.float64).reshape(-1, np.asarray(o).shape[-1]) for o in a]
        except Exception:
            return None
        return np.array([[r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()]
                         for r in rows], np.float32).reshape(-1, 4)
    a = a.astype(np.float64)
    if a.ndim == 3 and a.shape[-1] >= 2:                    # corners
        return np.stack([a[..., 0].min(1), a[..., 1].min(1), a[..., 0].max(1),
                         a[..., 1].max(1)], 1).astype(np.float32)
    if a.ndim == 2 and a.shape[1] == 4:
        b = a.astype(np.float32)
        fmt = box_format
        if fmt == "auto":
            fmt = "xyxy" if len(b) == 0 or np.all((b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])) else "xywh"
        if fmt == "xywh":
            b = b.copy()
            b[:, 2:] += b[:, :2]
        return b
    if a.ndim == 2 and a.shape[1] >= 6 and a.shape[1] % 2 == 0:   # flattened (x,y) corners
        c = a.reshape(len(a), -1, 2)
        return _boxes_from_array(c)
    if a.ndim == 2 and len(a) == 0:
        return np.zeros((0, 4), np.float32)
    return None


def extract_boxes_2d(annos, camera, box_format="auto"):
    """2D boxes for one camera from one frame of label.pkl 'annos'.
    DSERT-RoLL stores them as annos['gt_boxes_event' | 'gt_boxes_thermal' |
    'gt_boxes_rgb'] with class names in annos['name']. The per-box layout is
    handled by _boxes_from_array(); `cache --peek` prints a real example.
    Returns (boxes xyxy float32 (K,4) in that camera's native pixels, names) or None.
    After changing this, run `python run_experiments.py labels` (no download)."""
    if not isinstance(annos, dict):
        return None
    cam = camera.split("_")[0].lower()                       # event / thermal / rgb
    key = next((k for k in annos if str(k).lower() == f"gt_boxes_{cam}"), None)
    if key is None:                                          # fallback: any 2d/bbox key
        key = next((k for k in annos if cam in str(k).lower() and
                    any(w in str(k).lower() for w in ("box", "2d"))), None)
    if key is None:
        return None
    boxes = _boxes_from_array(annos[key], box_format)
    if boxes is None:
        return None
    names = None
    for k in ("name", "names", "class", "classes", "category", "label", "labels", "type"):
        if k in annos and len(np.asarray(annos[k]).reshape(-1)) == len(boxes):
            names = [str(n) for n in np.asarray(annos[k]).reshape(-1)]
            break
    return boxes, names or ["object"] * len(boxes)


# ------------------------------------------------------------- alignment
CALIB_NAME = {"event": "Event_L", "thermal": "Thermal_L", "rgb": "RGB_L"}


def homography_to_event(calib, dense, out_hw):
    """3x3 H mapping NATIVE dense-camera pixels -> event-camera pixels at out_hw.
    Infinite-plane approximation as in DSERT-RoLL supp. Sec. 14:
        H = S_out @ K_event @ R_event<-dense @ inv(K_dense)
    Each camera's 'extrinsic' is taken as camera -> RGB_L (RGB_L has none, so it
    is the reference): R_event<-dense = R_event^T @ R_dense. The rotations are
    tiny (< 0.5 deg), so even if that direction is reversed the error is a few
    pixels; check the overlay that `inspect` saves."""
    def KR(name):
        c = calib[name]
        R = np.asarray(c["extrinsic"])[:3, :3] if "extrinsic" in c else np.eye(3)
        return np.asarray(c["intrinsic"], np.float64), R, np.asarray(c["shape"])  # shape = (W, H)
    Kd, Rd, _ = KR(CALIB_NAME[dense])
    Ke, Re, (We, He) = KR("Event_L")
    S = np.diag([out_hw[1] / We, out_hw[0] / He, 1.0])
    return S @ Ke @ Re.T @ Rd @ np.linalg.inv(Kd)


def warp_to_event(img, H, out_hw):
    """Warp a native dense image (H,W) float or (H,W,3) uint8 with H into the event plane."""
    from PIL import Image
    M = np.linalg.inv(H)
    M = M / M[2, 2]
    coeffs = tuple(M.reshape(-1)[:8])
    size = (out_hw[1], out_hw[0])

    def one(ch):
        im = Image.fromarray(np.ascontiguousarray(ch, np.float32))
        return np.asarray(im.transform(size, Image.PERSPECTIVE, coeffs, Image.BILINEAR), np.float32)
    if img.ndim == 3:
        return np.stack([one(img[..., c]) for c in range(img.shape[2])], -1).clip(0, 255).astype(np.uint8)
    return one(img)


def cache_key(args):
    import hashlib
    cfg = [args.cache_size, args.frame_step, args.frames_per_seq, args.event_bins, args.dense]
    return hashlib.md5(json.dumps(cfg).encode()).hexdigest()[:8]


def cache_paths(cache_dir, seq_rel, key, dense):
    w, s = seq_rel.split("/", 1)
    base = os.path.join(cache_dir, f"{w}__{s}")
    return dict(event=f"{base}__event_l__{key}.npy", dense=f"{base}__{dense}_l__{key}.npy",
                frames=os.path.join(cache_dir, "meta", f"{w}__{s}__{key}.json"),
                label_pkl=os.path.join(cache_dir, "raw_labels", f"{w}__{s}.pkl"),
                labels=os.path.join(cache_dir, "labels", f"{w}__{s}__{key}.json"))


def build_labels_for(paths, box_camera, box_format="auto"):
    """raw label.pkl + cached frame list -> labels json with boxes in CACHE pixels."""
    import pickle
    import warnings
    warnings.filterwarnings("ignore", message=".*numpy.core.*")
    with open(paths["frames"]) as fh:
        meta = json.load(fh)
    with open(paths["label_pkl"], "rb") as fh:
        info = pickle.load(fh)["info"]
    src_hw = meta["native_hw"].get(box_camera)
    Ho, Wo = meta["cache_hw"]
    frames, found, unread = [], 0, 0
    for fi in meta["frame_positions"]:
        r = extract_boxes_2d(info[fi].get("annos"), box_camera, box_format)
        if r is None or src_hw is None:
            frames.append(dict(boxes=[], classes=[]))
            unread += 1
            continue
        b, names = r
        found += len(b) > 0
        b = b * np.array([Wo / src_hw[1], Ho / src_hw[0]] * 2, np.float32)
        frames.append(dict(boxes=b.round(2).tolist(), classes=names))
    os.makedirs(os.path.dirname(paths["labels"]), exist_ok=True)
    with open(paths["labels"], "w") as fh:
        json.dump(dict(box_camera=box_camera, cache_hw=[Ho, Wo], frames=frames), fh)
    return found, len(frames), unread


def _label_msg(r):
    found, n, unread = r
    msg = f"objects in {found}/{n} frames"
    return msg + (f" ({unread} frames' box layout NOT recognised)" if unread else "")


def load_cached_labels(cache_dir, sequences):
    """Reads <cache>/labels/*.json -> same structure as load_coco_labels()."""
    files = glob.glob(os.path.join(cache_dir, "labels", "*.json"))
    raw, names = {}, set()
    for f in files:
        seq = next((s for s in sequences if s in os.path.basename(f)), None)
        if seq is None:
            continue
        with open(f) as fh:
            d = json.load(fh)
        raw[seq] = d
        for fr in d["frames"]:
            names.update(fr["classes"])
    order = sorted(names)
    cid = {n: i + 1 for i, n in enumerate(order)}
    labels = {}
    for seq, d in raw.items():
        h, w = d["cache_hw"]
        labels[seq] = [dict(boxes=np.array(fr["boxes"], np.float32).reshape(-1, 4),
                            labels=np.array([cid[c] for c in fr["classes"]], np.int64),
                            size=(h, w)) for fr in d["frames"]]
    if labels and not any(len(f["boxes"]) for v in labels.values() for f in v):
        print("[warn] cached label files exist but contain no boxes: run `cache --peek` "
              "and adjust extract_boxes_2d(), then `labels`")
        return {}, []
    return labels, order


def _stratified(seqs, k, seed):
    by_w = defaultdict(list)
    for s in seqs:
        by_w[s.split("/")[0]].append(s)
    rng = random.Random(seed)
    return [s for w in sorted(by_w) for s in rng.sample(by_w[w], min(k, len(by_w[w]))) ]


def dense_native(x, mod):
    """Decoded sensor data -> native-resolution (H,W) float32 thermal or (H,W,3) uint8 rgb."""
    if isinstance(x, dict):
        x = max((v for v in x.values() if getattr(v, "ndim", 0) >= 2), key=lambda v: v.size)
    x = np.asarray(x)
    if mod == "rgb":
        return x[..., :3].astype(np.uint8)
    x = x.astype(np.float32)
    if x.ndim == 3:
        x = x.mean(axis=0 if (x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3)) else -1)
    return x


def cmd_cache(args):
    import pickle
    import warnings
    from concurrent.futures import ThreadPoolExecutor
    from dsert_stream import DsertStream, SENSORS, REPO, _decode
    from huggingface_hub import HfFileSystem
    warnings.filterwarnings("ignore", message=".*numpy.core.*")     # old pickles, harmless
    fs = HfFileSystem()
    dkey = SENSOR_KEY[args.dense]
    helper = DsertStream(sequences=[], sensors=["event_l", dkey])
    Ho, Wo = args.cache_size

    def read_split(name):
        txt = helper._read(fs, f"{REPO}/split/{name}.txt").decode()
        return [l.strip().strip("/") for l in txt.splitlines() if l.strip()]

    def fetch_files(seq, frame, sensors, counter=None):
        out = {}
        for s in sensors:
            rel = frame["sensor"].get(SENSORS[s])
            if not rel:
                return None
            p = helper._resolve(seq, rel)
            b = helper._read(fs, p)
            if counter is not None:
                counter.append(len(b))
            out[s] = _decode(p, b)
        return out

    def dense_frame(d, H):
        nat = dense_native(d, args.dense)
        if H is not None:
            out = warp_to_event(nat, H, (Ho, Wo))
        elif args.dense == "rgb":
            out = np.stack([_resize(nat[..., c], (Ho, Wo)) for c in range(3)], -1).clip(0, 255).astype(np.uint8)
        else:
            out = _resize(nat, (Ho, Wo))
        return (out.astype(np.float16) if args.dense == "thermal" else out), list(nat.shape[:2])

    splits = {sp: read_split(sp) for sp in ("train", "val")}
    if args.peek:
        seq = splits["val"][0]
        label = pickle.loads(helper._read(fs, f"{REPO}/{seq}/label.pkl"))
        info = label["info"]
        n_obj = [len(np.asarray(f.get("annos", {}).get("name", []))) for f in info]
        fi = int(np.argmax(n_obj))
        print(f"{seq}: {len(info)} frames, objects per frame min {min(n_obj)} max {max(n_obj)}")
        print(f"\n--- annos of frame {fi} ({n_obj[fi]} objects) ---")
        describe(info[fi]["annos"], maxd=5)
        for k, v in info[fi]["annos"].items():
            if str(k).startswith("gt_boxes") and len(np.asarray(v)):
                print(f"  {k}[:2] =\n{np.asarray(v)[:2]}")
        for cam in ("event_l", "thermal_l", "rgb_l"):
            r = extract_boxes_2d(info[fi]["annos"], cam, args.box_format)
            print(f"extract_boxes_2d({cam}) ->", "None (layout not recognised)" if r is None else
                  f"{len(r[0])} boxes, first {np.round(r[0][:2], 1).tolist()}, classes {r[1][:4]}")
        calib = label["meta"]["calibration"]
        H = homography_to_event(calib, args.dense, (Ho, Wo))
        Wd, Hd = np.asarray(calib[CALIB_NAME[args.dense]]["shape"])
        corners = np.array([[0, 0, 1], [Wd, 0, 1], [Wd, Hd, 1], [0, Hd, 1]], float).T
        m = H @ corners
        print(f"\n{args.dense} image corners land in the {Ho}x{Wo} event plane at:",
              np.round((m[:2] / m[2]).T, 1).tolist())
        return

    if args.all_sequences:
        chosen = {sp: list(v) for sp, v in splits.items()}
    else:
        chosen = {"train": _stratified(splits["train"], args.per_weather[0], args.seed),
                  "val": _stratified(splits["val"], args.per_weather[1], args.seed)}
    os.makedirs(args.cache, exist_ok=True)
    sp_path = os.path.join(args.cache, "splits.json")
    old = json.load(open(sp_path)) if os.path.exists(sp_path) else {"train": [], "val": []}
    for sp in ("train", "val"):
        old[sp] = sorted(set(old[sp]) | set(chosen[sp]))
    with open(sp_path, "w") as fh:
        json.dump(old, fh, indent=1)
    key = cache_key(args)
    align = not args.no_align
    def interleave(seqs):            # round-robin over weather, so a size cap
        by_w = defaultdict(list)     # cuts every condition evenly
        for q in seqs:
            by_w[q.split("/")[0]].append(q)
        lists = [by_w[w] for w in sorted(by_w)]
        return [l[i] for i in range(max(map(len, lists), default=0)) for l in lists if i < len(l)]

    # val first (so it is complete), then train, both spread across weather
    todo = [("val", q) for q in interleave(chosen["val"])] + \
           [("train", q) for q in interleave(chosen["train"])]

    def cache_gb():
        tot = 0
        for root, _, files in os.walk(args.cache):
            tot += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        return tot / 2 ** 30
    print(f"caching {len(todo)} sequences -> {args.cache} (config {key}, {Ho}x{Wo}, every "
          f"{args.frame_step}th frame, up to {args.frames_per_seq}/sequence, "
          f"{args.dense} {'aligned to event camera' if align else 'NOT aligned'})")
    for k, (sp, seq) in enumerate(todo, 1):
        P = cache_paths(args.cache, seq, key, args.dense)
        tag = f"[{k}/{len(todo)}] {seq}"
        done = all(os.path.exists(P[x]) for x in ("event", "dense", "frames", "label_pkl"))
        if args.max_gb and not done and cache_gb() >= args.max_gb:
            print(f"\nstopping: cache is {cache_gb():.2f} GB (limit --max-gb {args.max_gb}). "
                  f"{len(todo) - k + 1} sequences not cached.")
            break
        if done:
            with open(P["frames"]) as fh:
                meta = json.load(fh)
            if meta.get("dense_aligned", False) != align:          # redo dense only
                t0 = time.time()
                with open(P["label_pkl"], "rb") as fh:
                    lab = pickle.load(fh)
                H = homography_to_event(lab["meta"]["calibration"], args.dense, (Ho, Wo)) if align else None
                frames = [lab["info"][i] for i in meta["frame_positions"]]
                with ThreadPoolExecutor(args.threads) as pool:
                    got = list(pool.map(lambda f: fetch_files(seq, f, [dkey]), frames))
                arr = np.stack([dense_frame(g[dkey], H)[0] for g in got])
                np.save(P["dense"] + ".tmp.npy", arr)
                os.replace(P["dense"] + ".tmp.npy", P["dense"])
                meta["dense_aligned"] = align
                with open(P["frames"], "w") as fh:
                    json.dump(meta, fh)
                print(f"{tag}: {args.dense} re-{'aligned' if align else 'resized'} "
                      f"({len(arr)} frames, {time.time() - t0:.0f}s)")
            print(f"{tag}: cached, {_label_msg(build_labels_for(P, args.box_camera, args.box_format))}")
            continue
        t0 = time.time()
        raw = helper._read(fs, f"{REPO}/{seq}/label.pkl")
        os.makedirs(os.path.dirname(P["label_pkl"]), exist_ok=True)
        with open(P["label_pkl"], "wb") as fh:
            fh.write(raw)
        lab = pickle.loads(raw)
        info = lab["info"]
        H = homography_to_event(lab["meta"]["calibration"], args.dense, (Ho, Wo)) if align else None
        positions = list(range(0, len(info), args.frame_step))[: args.frames_per_seq]
        downloaded = [len(raw)]
        ev_l, dn_l, kept, native = [], [], [], {}
        with ThreadPoolExecutor(args.threads) as pool:
            got = pool.map(lambda fi: fetch_files(seq, info[fi], ["event_l", dkey], downloaded), positions)
            for fi, data in zip(positions, got):
                if data is None:
                    continue
                x, y, t, pol = parse_events(data["event_l"])
                if args.event_hw:
                    shw = tuple(args.event_hw)
                else:   # calibration says 1152 x 704; infer only if it is missing
                    c = lab["meta"].get("calibration", {}).get("Event_L", {})
                    shw = ((int(c["shape"][1]), int(c["shape"][0])) if "shape" in c else
                           native.get("event_l") or (int(math.ceil((y.max() + 1) / 16) * 16),
                                                     int(math.ceil((x.max() + 1) / 16) * 16)))
                native["event_l"] = list(shw)
                ev_l.append(voxel_grid_fast(x, y, t, pol, shw, (Ho, Wo), args.event_bins)
                            .astype(np.float16))
                dd, native[dkey] = dense_frame(data[dkey], H)
                dn_l.append(dd)
                kept.append(fi)
        if not kept:
            print(f"{tag}: no usable frames, skipped")
            continue
        for name, arr in (("event", np.stack(ev_l)), ("dense", np.stack(dn_l))):
            np.save(P[name] + ".tmp.npy", arr)
            os.replace(P[name] + ".tmp.npy", P[name])
        os.makedirs(os.path.dirname(P["frames"]), exist_ok=True)
        with open(P["frames"], "w") as fh:
            json.dump(dict(sequence=seq, split=sp, frame_positions=kept, cache_hw=[Ho, Wo],
                           native_hw=native, frame_step=args.frame_step, dense_aligned=align), fh)
        lm = _label_msg(build_labels_for(P, args.box_camera, args.box_format))
        mb = (os.path.getsize(P["event"]) + os.path.getsize(P["dense"])) / 2 ** 20
        print(f"{tag} ({sp}): {len(kept)} frames, {mb:.0f} MB on disk, "
              f"{sum(downloaded) / 2 ** 20:.0f} MB downloaded, "
              f"{lm}, {time.time() - t0:.0f}s")


def cmd_labels(args):
    """Rebuild <cache>/labels/*.json from saved label.pkl files (no download)."""
    metas = glob.glob(os.path.join(args.cache, "meta", "*.json"))
    tot = 0
    for m in metas:
        with open(m) as fh:
            seq = json.load(fh)["sequence"]
        key = os.path.basename(m).rsplit("__", 1)[1][:-5]
        P = cache_paths(args.cache, seq, key, args.dense)
        r = build_labels_for(P, args.box_camera, args.box_format)
        tot += r[0]
        print(f"{seq}: {_label_msg(r)}")
    print(f"done: {len(metas)} sequences, {tot} frames with objects")


# ================================================================= commands
def cmd_inspect(args):
    groups = find_groups(args.cache, args.dense)
    all_files = glob.glob(os.path.join(args.cache, "*.npy"))
    bad = [f for f in all_files if not parse_cache_name(f)]
    print(f"{len(all_files)} .npy files, {len(bad)} with unrecognised names")
    print(f"{len(groups)} complete event+{args.dense} groups, "
          f"{len({g['seq'] for g in groups})} sequences")
    if not groups:
        no_cache_exit(args)
    by_cond = defaultdict(int)
    for g in groups:
        by_cond[g["cond"]] += 1
    print("groups per condition:", dict(by_cond))
    for g in groups[:3]:
        print(f"\n{g['cond']} {g['seq']} hash={g['hash']}")
        for mod in ("event", "dense"):
            a = np.load(g[mod], mmap_mode="r")
            lay = detect_layout(a.shape)
            fr = get_frame(a, 0, lay)
            vals, counts = np.unique(fr.reshape(fr.shape[0], -1).T, axis=0, return_counts=True)
            top = vals[counts.argmax()]
            print(f"  {mod:6s} shape={a.shape} dtype={a.dtype} layout={lay} "
                  f"frames={num_frames(a.shape, lay)} frame0 C,H,W={fr.shape} "
                  f"min={fr.min():.3g} max={fr.max():.3g} nonzero={np.mean(fr != 0):.1%}\n"
                  f"         per-channel mean={[round(float(v), 1) for v in fr.mean((1, 2))]} "
                  f"most common pixel={[round(float(v), 1) for v in top]} "
                  f"({counts.max() / counts.sum():.0%} of pixels)")
    try:                                              # look at the data, don't guess
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.join(args.out, "inspect"), exist_ok=True)
        for k, g in enumerate(groups[:4]):
            ae_, ad_ = (np.load(g[m], mmap_mode="r") for m in ("event", "dense"))
            le, ld = detect_layout(ae_.shape), detect_layout(ad_.shape)
            fig, axs = plt.subplots(3, 3, figsize=(12, 6.4))
            n = min(num_frames(ae_.shape, le), num_frames(ad_.shape, ld))
            for col, f in enumerate(sorted({0, n // 2, n - 1})):
                for row, (a, lay, nm) in enumerate([(ae_, le, "event"), (ad_, ld, args.dense)]):
                    fr = get_frame(a, f, lay)
                    if nm == "event" and fr.shape[0] > 3:      # voxel grid: show |events|
                        img = np.abs(fr).sum(0)
                        axs[row, col].imshow(img, cmap="magma", vmin=0,
                                             vmax=max(np.percentile(img, 99.5), 1e-6))
                    else:
                        img = fr[0] if fr.shape[0] != 3 else np.moveaxis(fr, 0, -1) / max(fr.max(), 1)
                        axs[row, col].imshow(img, cmap="gray")
                    axs[row, col].set_title(f"{nm} frame {f}", fontsize=9)
                    axs[row, col].axis("off")
                fe, fd = get_frame(ae_, f, le), get_frame(ad_, f, ld)
                if fe.shape[1:] == fd.shape[1:]:                   # alignment check
                    gray = fd[0] if fd.shape[0] != 3 else fd.mean(0)
                    gray = (gray - gray.min()) / max(gray.max() - gray.min(), 1e-6)
                    ev = np.abs(fe).sum(0)
                    ev = np.clip(ev / max(np.percentile(ev, 99), 1e-6), 0, 1)
                    axs[2, col].imshow(np.stack([np.maximum(gray, ev), gray * (1 - ev),
                                                 gray * (1 - ev)], -1))
                axs[2, col].set_title("overlay: events in red", fontsize=9)
                axs[2, col].axis("off")
            fig.suptitle(f"{g['cond']} {g['seq']} {g['hash']}", fontsize=10)
            p = os.path.join(args.out, "inspect", f"sample_{k}.png")
            fig.savefig(p, dpi=110, bbox_inches="tight")
            plt.close(fig)
        print(f"\nsaved sample frames to {args.out}/inspect/")
    except ImportError:
        print("\n(matplotlib not installed: no sample images)")
    train, val = split_sequences(groups, seed=args.seed, cache_dir=args.cache)
    print(f"\nsplit: {len(train)} train / {len(val)} val sequences: val = {val}")
    total = sum(num_frames(np.load(g['event'], mmap_mode='r').shape,
                           detect_layout(np.load(g['event'], mmap_mode='r').shape)) for g in groups)
    print(f"total cached frames (event+{args.dense} pairs): {total}")
    jsons = glob.glob(os.path.join(args.dataset, "**", "*.json"), recursive=True)
    print(f"\n{len(jsons)} JSON files under {args.dataset}; first few:")
    for j in jsons[:5]:
        print("  ", j)
    labels, names = load_labels(args, sorted({g["seq"] for g in groups}))
    print(f"labels matched for {len(labels)} sequences; classes: {names}")
    for seq, fr in list(labels.items())[:3]:
        nb = sum(len(f["boxes"]) for f in fr)
        print(f"  {seq}: {len(fr)} images, {nb} boxes, image size {fr[0]['size']}")
    if jsons and not labels:
        with open(jsons[0]) as fh:
            d = json.load(fh)
        print("\nNo match. Top-level keys of first JSON:",
              list(d)[:10] if isinstance(d, dict) else type(d))


def cmd_pretrain(args):
    dev = pick_device(args.device)
    groups, train, _, _, _ = setup(args)
    ds = CachedPairs(groups, train, args.dense, args.size)
    ce, cd = channels(ds)
    print(f"pretrain {args.model} on {len(ds)} frames, event ch={ce}, {args.dense} ch={cd}, "
          f"device={dev}")
    names = args.only or [k for k, v in EXPERIMENTS.items() if v is not None]
    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)
    for name in names:
        cfg = EXPERIMENTS[name]
        if cfg is None:
            continue
        ck = os.path.join(args.out, "ckpt", f"{name}.pt")
        if os.path.exists(ck) and not args.overwrite:
            print(f"skip {name} (exists)")
            continue
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        ae = build_model(args.model, cd, ce, imagenet=not args.no_imagenet, size=args.size).to(dev)
        # --encoder-lr-scale: encoders learn at lr*scale, new layers at lr.
        # 0 freezes the encoders completely (weights AND BatchNorm statistics),
        # so only fusion + decoders are pretrained.
        enc = [ae.enc_event, ae.enc_dense]
        enc_ids = {id(p) for m in enc for p in m.parameters()}
        rest = [p for p in ae.parameters() if id(p) not in enc_ids]
        enc_params = [p for m in enc for p in m.parameters()]
        if args.encoder_lr_scale == 0:
            for m in enc:
                m.requires_grad_(False)
            groups_ = [dict(params=rest, lr=args.lr)]
        else:
            groups_ = [dict(params=rest, lr=args.lr),
                       dict(params=enc_params, lr=args.lr * args.encoder_lr_scale)]
        opt = torch.optim.AdamW(groups_, lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
        save_at = sorted(set(s_ for s_ in (args.save_at or []) if 0 < s_ < args.steps))
        meta = dict(model=args.model, size=list(args.size), event_bins=ce, dense_ch=cd,
                    dense=args.dense, imagenet=not args.no_imagenet, cfg=cfg,
                    encoder_lr_scale=args.encoder_lr_scale)
        print(f"{name}: encoder lr x{args.encoder_lr_scale}"
              + (f", snapshots at steps {save_at}" if save_at else ""))
        it = infinite(train_loader(ds, args))
        ae.train()
        if args.encoder_lr_scale == 0:
            for m in enc:
                m.eval()                    # keep pretrained BatchNorm statistics
        t0, run = time.time(), 0.0
        for step in range(1, args.steps + 1):
            ev, dn, _ = next(it)
            run += train_step(ae, opt, ev.to(dev), dn.to(dev), cfg)
            sched.step()
            if step in save_at:             # snapshot mid-schedule (lr not yet decayed)
                snap = os.path.join(args.out, f"at_step{step}", "ckpt", f"{name}.pt")
                os.makedirs(os.path.dirname(snap), exist_ok=True)
                torch.save(dict(state=ae.state_dict(), steps=step, **meta), snap)
                print(f"saved snapshot {snap}")
            if step % args.log_every == 0:
                write_rows(os.path.join(args.out, "pretrain_loss.csv"),
                           [dict(experiment=name, step=step, loss=run / args.log_every)])
                print(f"{name} step {step}/{args.steps} loss {run / args.log_every:.4f} "
                      f"({time.time() - t0:.0f}s)")
                run = 0.0
        torch.save(dict(state=ae.state_dict(), steps=args.steps, **meta), ck)
        print(f"saved {ck}")


def cmd_recon(args):
    """How well does each pretrained checkpoint reconstruct the hidden modality
    on VALIDATION frames? Writes <out>/recon.csv and <out>/recon_<name>.png.
    skill = 1 - MSE(model) / MSE(predicting each frame's mean): 0 means no better
    than a flat guess, 1 is perfect. Also reported inside/outside labelled boxes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = pick_device(args.device)
    groups, _, val, labels, _ = setup(args, need_labels=True)
    ds = CachedPairs(groups, val, args.dense, args.size, labels)
    idx = list(range(len(ds)))[::max(1, len(ds) // args.max_frames)]
    names = args.only or sorted(os.path.splitext(f)[0] for f in
                                os.listdir(os.path.join(args.out, "ckpt")) if f.endswith(".pt"))
    rows = []
    for name in names:
        ck = os.path.join(args.out, "ckpt", f"{name}.pt")
        if not os.path.exists(ck):
            print(f"skip {name}: no checkpoint")
            continue
        saved = torch.load(ck, map_location="cpu")
        ae = build_model(saved.get("model", "crossmae"), saved["dense_ch"], saved["event_bins"],
                         imagenet=False, size=args.size)
        ae.load_state_dict(saved["state"])
        ae.to(dev).eval()
        acc = {h: defaultdict(float) for h in ("dense", "event")}
        vis = []
        with torch.no_grad():
            for k in range(0, len(idx), args.batch):
                batch = [ds[i] for i in idx[k:k + args.batch]]
                ev, dn, tg = collate(batch)
                ev, dn = ev.to(dev), dn.to(dev)
                box = torch.zeros(ev.shape[0], 1, *ev.shape[2:], device=dev)
                for j, t in enumerate(tg):
                    for b in (t["boxes"] if t is not None else []):
                        x0, y0, x1, y1 = [int(round(float(v))) for v in b]
                        box[j, :, max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 1
                for hide in ("dense", "event"):
                    pred = ae(ev, dn, hide)
                    tgt = dn if hide == "dense" else ev
                    err = (pred - tgt).pow(2).mean(1, keepdim=True)
                    flat = (tgt - tgt.mean(dim=(2, 3), keepdim=True)).pow(2).mean(1, keepdim=True)
                    a = acc[hide]
                    a["mse"] += err.sum().item(); a["base"] += flat.sum().item()
                    a["n"] += err.numel()
                    a["mse_in"] += (err * box).sum().item(); a["base_in"] += (flat * box).sum().item()
                    a["mse_out"] += (err * (1 - box)).sum().item()
                    a["base_out"] += (flat * (1 - box)).sum().item()
                    if hide == "dense" and len(vis) < args.n_vis:
                        dpred = pred.cpu()
                        epred = ae(ev, dn, "event").cpu()
                        for j in range(min(ev.shape[0], args.n_vis - len(vis))):
                            vis.append((ev[j].cpu(), dn[j].cpu(), dpred[j], epred[j]))
        for hide, a in acc.items():
            sk = lambda m, b: 1 - a[m] / a[b] if a[b] > 0 else float("nan")
            row = dict(experiment=name, hidden=hide, frames=len(idx),
                       pretrain_steps=saved.get("steps"), mse=a["mse"] / max(a["n"], 1),
                       skill=sk("mse", "base"), skill_in_boxes=sk("mse_in", "base_in"),
                       skill_outside=sk("mse_out", "base_out"))
            rows.append(row)
            print(f"{name:14s} reconstruct {hide:5s}: skill {row['skill']:.3f} "
                  f"(in boxes {row['skill_in_boxes']:.3f}, outside {row['skill_outside']:.3f})")
        if vis:
            fig, axs = plt.subplots(len(vis), 4, figsize=(13, 2.4 * len(vis)), squeeze=False)
            for r, (ev, dn, dp, ep) in enumerate(vis):
                es, eps = ev.sum(0), ep.sum(0)
                lim = float(max(es.abs().max(), 1e-6))
                panels = [(dn[0], "thermal (target)", "gray", None),
                          (dp[0], "thermal from events", "gray", None),
                          (es, "events (target)", "seismic", lim),
                          (eps, "events from thermal", "seismic", lim)]
                for c, (img, title, cmap, l) in enumerate(panels):
                    kw = dict(vmin=-l, vmax=l) if l else dict(vmin=float(dn[0].min()),
                                                               vmax=float(dn[0].max()))
                    axs[r, c].imshow(img.numpy(), cmap=cmap, **kw)
                    axs[r, c].set_axis_off()
                    if r == 0:
                        axs[r, c].set_title(title, fontsize=9)
            fig.suptitle(f"{name}: validation reconstructions", fontsize=10)
            fig.tight_layout()
            png = os.path.join(args.out, f"recon_{name}.png")
            fig.savefig(png, dpi=90)
            plt.close(fig)
            print(f"  figure: {png}")
    if rows:
        write_rows(os.path.join(args.out, "recon.csv"), rows)


def cmd_overlap(args):
    _, _, val, labels, _ = setup(args, need_labels=True)
    groups = find_groups(args.cache, args.dense)
    ds = CachedPairs(groups, val, args.dense, args.size, labels, require_labels=True)
    idx = ds.index[::max(1, len(ds.index) // args.max_frames)]
    modes = WEIGHT_MODES
    acc = {m: [] for m in modes}
    show = []
    for gi, f in idx:
        ev, dn, tgt = ds.get(gi, f)
        if tgt is None or len(tgt["boxes"]) == 0:
            continue
        boxes = tgt["boxes"].numpy()
        for m in modes:
            w = weight_map(ev[None], m)[0, 0].numpy()
            r = heatmap_box_overlap(w, boxes, *args.size)
            if r:
                acc[m].append(r)
        if len(show) < args.n_vis:
            show.append((ev, dn, boxes, ds.groups[gi]["cond"]))
    rows = []
    for m in modes:
        a = np.array(acc[m])
        if len(a):
            rows.append(dict(mode=m, frames=len(a), mass_in_boxes=a[:, 0].mean(),
                             box_area=a[:, 1].mean(), lift=a[:, 2].mean()))
            print(f"{m:8s} mass in boxes {a[:, 0].mean():.3f}  box area {a[:, 1].mean():.3f}"
                  f"  lift {a[:, 2].mean():.2f}  (n={len(a)})")
    if rows:
        write_rows(os.path.join(args.out, "overlap.csv"), rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        for k, (ev, dn, boxes, cond) in enumerate(show):
            fig, axs = plt.subplots(1, len(modes) + 1, figsize=(3.2 * (len(modes) + 1), 3))
            axs[0].imshow(dn[0], cmap="gray")
            axs[0].set_title(f"{args.dense} ({cond})")
            for ax, m in zip(axs[1:], modes):
                ax.imshow(weight_map(ev[None], m)[0, 0], cmap="magma")
                ax.set_title(m)
            for ax in axs:
                for x0, y0, x1, y1 in boxes:
                    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="c", lw=1))
                ax.axis("off")
            os.makedirs(os.path.join(args.out, "heatmaps"), exist_ok=True)
            fig.savefig(os.path.join(args.out, "heatmaps", f"frame_{k}.png"), dpi=110,
                        bbox_inches="tight")
            plt.close(fig)
        print(f"saved {len(show)} heatmap figures to {args.out}/heatmaps/")
    except ImportError:
        print("matplotlib not installed: skipped heatmap figures")


_no_grad = torch.no_grad() if torch else (lambda f: f)   # (Python < 3.9 compatible)


@_no_grad
def evaluate(det, ds, dev, num_classes, batch):
    """-> ({class: AP50}, stats) where stats describes what the detector output."""
    det.eval()
    dets, gts = [], []
    for ev, dn, tgts in DataLoader(ds, batch, collate_fn=collate):
        x = list(torch.cat([ev, dn], 1).to(dev))
        for out, t in zip(det(x), tgts):
            dets.append((out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(),
                         out["labels"].cpu().numpy()))
            gts.append((t["boxes"].numpy(), t["labels"].numpy()))
    det.train()
    n = max(len(dets), 1)
    stats = dict(dets_per_frame=sum(len(d[1]) for d in dets) / n,
                 mean_top_score=float(np.mean([d[1].max() if len(d[1]) else 0.0 for d in dets]))
                 if dets else 0.0)
    return {c: average_precision(dets, gts, c) for c in range(1, num_classes)}, stats


def box_size_report(ds, size):
    ws, hs = [], []
    for g, f in ds.index:
        lab = ds.labels[ds.groups[g]["seq"]][f]
        b = lab["boxes"] * np.array([size[1] / lab["size"][1], size[0] / lab["size"][0]] * 2)
        ws += list(b[:, 2] - b[:, 0])
        hs += list(b[:, 3] - b[:, 1])
    if not ws:
        return "no boxes"
    side = np.minimum(ws, hs)
    return (f"{len(ws)} boxes at {size[0]}x{size[1]}: median {np.median(ws):.0f}x{np.median(hs):.0f} px, "
            f"shorter side <16 px: {np.mean(side < 16):.0%}, <8 px: {np.mean(side < 8):.0%}")


def cmd_probe(args):
    dev = pick_device(args.device)
    groups, train, val, labels, names = setup(args, need_labels=True)
    tr = CachedPairs(groups, train, args.dense, args.size, labels, require_labels=True)
    va = CachedPairs(groups, val, args.dense, args.size, labels, require_labels=True)
    va_idx = va.index[::max(1, len(va.index) // args.max_val)]
    va_sub = Subset(va, va_idx)
    ce, cd = channels(tr)
    ncls = len(names) + 1
    print(f"probe: {len(tr)} labelled train frames, {len(va_sub)} val frames, classes {names}")
    print("train", box_size_report(tr, args.size))
    old_csv = os.path.join(args.out, "probe.csv")
    if os.path.exists(old_csv):
        with open(old_csv) as fh:
            header = fh.readline()
        if "train_mAP50" not in header:          # results from the old detector setup
            os.replace(old_csv, os.path.join(args.out, "probe_old_setup.csv"))
            print("moved previous probe.csv to probe_old_setup.csv (different detector setup)")
    print(f"settings: model {args.model}, score_thresh {args.score_thresh}, encoders "
          f"{'fine-tuned' if args.finetune else 'frozen'}, fusion "
          f"{'trained' if (args.train_fusion or args.finetune) else 'frozen'}, {args.steps} steps, "
          f"head with {args.head_convs} conv layers per branch, lr {args.lr}, "
          f"warmup {args.warmup} steps, grad clip {args.clip}, batch {args.batch}")
    runs = args.only or list(EXPERIMENTS)
    if "a_imagenet" not in runs and not args.only:
        runs = ["a_imagenet"] + runs
    fracs = args.fractions or LABEL_FRACTIONS
    for name in runs:
        for frac in fracs:
            torch.manual_seed(args.seed)
            if name == "a_random_init":
                ae = build_model(args.model, cd, ce, imagenet=False, size=args.size)
            elif name == "a_imagenet":
                ae = build_model(args.model, cd, ce, imagenet=True, size=args.size)
            else:
                ck = os.path.join(args.out, "ckpt", f"{name}.pt")
                if not os.path.exists(ck):
                    print(f"skip {name}: no checkpoint (run pretrain first)")
                    break
                saved = torch.load(ck, map_location="cpu")
                if saved.get("model", "crossmae") != args.model:
                    raise SystemExit(f"{ck} was pretrained with --model {saved.get('model', 'crossmae')}"
                                     f", but probe is running --model {args.model}. Use the same "
                                     "--model, or a different --out folder.")
                ae = build_model(args.model, saved["dense_ch"], saved["event_bins"],
                                 imagenet=False, size=args.size)
                ae.load_state_dict(saved["state"])
            det = build_detector(ae, num_classes=ncls, dense_ch=cd, event_bins=ce,
                                 size=args.size, freeze=not args.finetune,
                                 score_thresh=args.score_thresh,
                                 train_fusion=args.train_fusion,
                                 head_convs=args.head_convs).to(dev)
            sub_idx = label_subset(tr.index, frac, seed=args.seed)
            sub = Subset(tr, sub_idx)
            it = infinite(train_loader(sub, args))
            params = [p for p in det.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
            # linear warmup over --warmup steps, then cosine decay to 0.
            # With --warmup 0 this is identical to the old CosineAnnealingLR.
            wu, total = args.warmup, args.steps
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: (s + 1) / wu if s < wu else
                0.5 * (1 + math.cos(math.pi * (s - wu) / max(1, total - wu))))
            det.train()
            t0 = time.time()
            diverged = None
            for step in range(1, args.steps + 1):          # same budget for every run
                ev, dn, tgts = next(it)
                x = list(torch.cat([ev, dn], 1).to(dev))
                tg = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
                parts = det(x, tg)
                loss = sum(parts.values())
                if not torch.isfinite(loss):               # NaN/inf: weights would be ruined
                    diverged = step
                    detail = " ".join(f"{k.replace('bbox_', '')} {v.item():.3f}"
                                      for k, v in parts.items())
                    print(f"{name} {frac:.0%} step {step}/{args.steps} loss {loss.item()} "
                          f"[{detail}] DIVERGED: stopping this run")
                    break
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, args.clip)
                opt.step()
                sched.step()
                if step == 1 or step % args.log_every == 0:
                    detail = " ".join(f"{k.replace('bbox_', '')} {v.item():.3f}"
                                      for k, v in parts.items())
                    print(f"{name} {frac:.0%} step {step}/{args.steps} loss {loss.item():.3f} "
                          f"[{detail}] ({time.time() - t0:.0f}s)")
            settings = dict(lr=args.lr, warmup=args.warmup, clip=args.clip, batch=args.batch,
                            steps=args.steps, head_convs=args.head_convs,
                            train_fusion=int(bool(args.train_fusion or args.finetune)),
                            diverged_step=diverged if diverged else "")
            if diverged:   # nothing meaningful to evaluate: record zeros, as before
                ap = {c: 0.0 for c in range(1, ncls)}
                tr_ap = dict(ap)
                st = dict(dets_per_frame=0.0, mean_top_score=0.0)
            else:
                ap, st = evaluate(det, va_sub, dev, ncls, args.batch)
                # same metric on (up to 200) TRAINING frames: if this is also ~0 the
                # head is not learning at all; if only val is ~0 it is overfitting
                tr_ap, _ = evaluate(det, Subset(tr, sub_idx[::max(1, len(sub_idx) // 200)]),
                                    dev, ncls, args.batch)
            mean = lambda d: float(np.mean([v for v in d.values() if not math.isnan(v)] or [float("nan")]))
            row = dict(experiment=name, fraction=frac, train_frames=len(sub), seed=args.seed,
                       mAP50=mean(ap), train_mAP50=mean(tr_ap),
                       dets_per_frame=round(st["dets_per_frame"], 1),
                       mean_top_score=round(st["mean_top_score"], 3),
                       **{f"AP50_{names[c - 1]}": v for c, v in ap.items()}, **settings)
            write_rows(os.path.join(args.out, "probe.csv"), [row])
            print(f"==> {name} @ {frac:.0%}: val mAP50 {row['mAP50']:.3f}  "
                  f"train mAP50 {row['train_mAP50']:.3f}  "
                  f"({row['dets_per_frame']} detections/frame, mean top score {row['mean_top_score']})")


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = os.path.join(args.out, "probe.csv")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    res = defaultdict(lambda: defaultdict(list))
    for r in rows:
        res[r["experiment"]][float(r["fraction"])].append(float(r["mAP50"]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in sorted(res):
        fr = sorted(res[name])
        m = [np.mean(res[name][f]) for f in fr]
        s = [np.std(res[name][f]) for f in fr]
        style = "--" if name.startswith("a_") else "-"
        ax.errorbar([100 * f for f in fr], m, yerr=s, fmt=style + "o", capsize=3, label=name)
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100], ["10%", "25%", "50%", "100%"])
    ax.set_xlabel("Labelled training data")
    ax.set_ylabel("mAP @ IoU 0.5")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    out = os.path.join(args.out, "label_efficiency.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    lo = min({float(r["fraction"]) for r in rows})
    print(f"\n2x2 (mAP50 at {lo:.0%} labels):")
    for d in ("e2d", "d2e"):
        cells = [np.mean(res[f"{d}_{w}"][lo]) if res[f"{d}_{w}"][lo] else float("nan")
                 for w in ("plain", "weighted")]
        print(f"  {d}: plain {cells[0]:.3f}   weighted {cells[1]:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hftoken", default="",
                   help="Hugging Face token (optional). Simpler: set the HF_TOKEN environment "
                        "variable, which huggingface_hub reads automatically")
    p.add_argument("command", choices=["cache", "labels", "inspect", "overlap", "pretrain",
                                       "probe", "plot", "recon"])
    p.add_argument("--model", default="crossmae", choices=MODELS,
                   help="model to pretrain/probe: crossmae (ResNet-18) or videomae")
    p.add_argument("--cache", default="dsert_frames_cache")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--dense", default="thermal", choices=["thermal", "rgb"])
    p.add_argument("--size", type=int, nargs=2, default=[256, 384], metavar=("H", "W"),
                   help="model input size (match --cache-size)")
    p.add_argument("--out", default="results")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--steps", type=int, default=None, help="pretrain 5000 / probe 3000")
    p.add_argument("--lr", type=float, default=None, help="pretrain 3e-4 / probe 1e-3")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--only", nargs="*", help="experiment names to run")
    p.add_argument("--fractions", type=float, nargs="*")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-imagenet", action="store_true", help="pretrain from random init")
    p.add_argument("--finetune", action="store_true", help="unfreeze encoder in probe")
    p.add_argument("--train-fusion", action="store_true",
                   help="probe: keep encoders frozen but train the fusion layers")
    p.add_argument("--head-convs", type=int, default=4,
                   help="probe: conv layers per FCOS head branch (4 default, 1 shallow, 0 minimal)")
    p.add_argument("--encoder-lr-scale", type=float, default=1.0,
                   help="pretrain: encoder lr = lr x this (0 = encoders frozen, incl. BatchNorm)")
    p.add_argument("--save-at", type=int, nargs="*", default=None,
                   help="pretrain: also save snapshots at these steps, to <out>/at_step<N>/ckpt/")
    p.add_argument("--warmup", type=int, default=0,
                   help="probe: linear learning-rate warmup steps before the cosine decay")
    p.add_argument("--clip", type=float, default=10.0,
                   help="probe: max gradient norm (lower = gentler updates)")
    p.add_argument("--score-thresh", type=float, default=0.05,
                   help="probe: minimum detection score kept for evaluation")
    p.add_argument("--max-frames", type=int, default=500, help="overlap: frames to sample")
    p.add_argument("--max-val", type=int, default=1000, help="probe: val frames to score")
    p.add_argument("--n-vis", type=int, default=6, help="overlap: figures to save")
    c = p.add_argument_group("cache (download from Hugging Face)")
    c.add_argument("--peek", action="store_true", help="print label.pkl / sensor structure only")
    c.add_argument("--per-weather", type=int, nargs=2, default=[4, 2], metavar=("TRAIN", "VAL"),
                   help="sequences per weather condition from the official split")
    c.add_argument("--all-sequences", action="store_true", help="cache the whole split")
    c.add_argument("--frame-step", type=int, default=5, help="keep every Nth frame")
    c.add_argument("--frames-per-seq", type=int, default=100)
    c.add_argument("--cache-size", type=int, nargs=2, default=[256, 384], metavar=("H", "W"),
                   help="RESOLUTION frames are stored at (height width), not a storage "
                        "limit. ~1.1 MB/frame at 256x384; doubling both sides = ~4x")
    c.add_argument("--max-gb", type=float, default=None,
                   help="stop caching new sequences once the cache folder reaches this "
                        "size (it can overshoot by one sequence, ~30-40 MB)")
    c.add_argument("--event-bins", type=int, default=5)
    c.add_argument("--event-hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="event sensor size (default: inferred from coordinates)")
    c.add_argument("--box-camera", default="event_l",
                   help="whose 2D boxes to keep (event_l, thermal_l, rgb_l)")
    c.add_argument("--box-format", default="auto", choices=["auto", "xyxy", "xywh"])
    c.add_argument("--threads", type=int, default=16)
    c.add_argument("--no-align", action="store_true",
                   help="don't warp the dense camera into the event camera's view")
    a = p.parse_args()
    if a.hftoken:                       # don't overwrite an existing HF_TOKEN with ""
        os.environ["HF_TOKEN"] = a.hftoken
    if a.steps is None:
        a.steps = 3000 if a.command == "probe" else 5000
    if a.lr is None:
        a.lr = 1e-3 if a.command == "probe" else 3e-4
    a.size = tuple(a.size)
    a.cache_size = list(a.cache_size)
    dict(cache=cmd_cache, labels=cmd_labels, inspect=cmd_inspect, overlap=cmd_overlap, pretrain=cmd_pretrain,
         probe=cmd_probe, plot=cmd_plot, recon=cmd_recon)[a.command](a)


if __name__ == "__main__":
    main()