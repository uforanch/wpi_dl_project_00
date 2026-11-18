"""
Two-stream cross-modal autoencoder for DSERT-RoLL (event + RGB/thermal),
following the professor's scoped-down design.

  Stage 1  label-free pretraining
           event encoder + dense encoder -> fusion bottleneck (one stream hidden
           by a learned mask token) -> decoder rebuilds the hidden modality
           -> loss weighted by a map computed from event activity
  Stage 2  label-efficiency evaluation
           frozen encoders + fusion -> torchvision FCOS head
           -> 2D AP@0.5 at 10/25/50/100 % of labels

Only PyTorch + torchvision. UNTESTED: written without being able to install
torch in this environment -- run the __main__ smoke test first.

Input sizes: H and W must be divisible by 16 (encoder stride), and by 32 for
FCOS batching. E.g. crop/resize everything to 512 x 640.
"""
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.detection import FCOS
from torchvision.models.detection.anchor_utils import AnchorGenerator

EVENT_BINS = 5      # same as DSERT-RoLL
DENSE_CH = 3        # 3 = RGB, 1 = thermal
FEAT = 256          # ResNet-18 layer3 width


# =============================================================== preprocessing
def event_to_rgb_homography(K_rgb, K_evt, R_evt_to_rgb):
    """DSERT-RoLL supp. Sec. 14: H = K_rgb @ R @ inv(K_evt).
    Rescale K_rgb to your target resolution first. Treat as approximate."""
    return K_rgb @ R_evt_to_rgb @ np.linalg.inv(K_evt)


def events_to_voxel_grid(x, y, t, p, H_mat=None, bins=EVENT_BINS, height=512, width=640):
    """Raw events (numpy 1-D arrays) -> torch (bins, height, width) voxel grid.
    If H_mat is given, event pixel coords are warped onto the RGB image plane
    BEFORE voxelising (avoids the grid artefacts of warping a finished image).
    Polarity is split between the two nearest time bins (Zhu et al. 2019)."""
    x, y, t = x.astype(np.float64), y.astype(np.float64), t.astype(np.float64)
    if H_mat is not None:
        pts = H_mat @ np.stack([x, y, np.ones_like(x)])
        x, y = pts[0] / pts[2], pts[1] / pts[2]
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    keep = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    xi, yi, t, p = xi[keep], yi[keep], t[keep], p[keep]
    grid = np.zeros((bins, height, width), np.float32)
    if len(t) == 0:
        return torch.from_numpy(grid)
    pol = np.where(p > 0, 1.0, -1.0)
    tn = (t - t.min()) / max(t.max() - t.min(), 1e-9) * (bins - 1)
    t0 = np.floor(tn).astype(int)
    for b, w in ((t0, 1 - (tn - t0)), (t0 + 1, tn - t0)):
        ok = b < bins
        np.add.at(grid, (b[ok], yi[ok], xi[ok]), pol[ok] * w[ok])
    g = torch.from_numpy(grid)
    return g / (g.abs().std() + 1e-6)          # rough scale normalisation


# ==================================================================== encoders
def resnet18_encoder(in_ch, imagenet=True):
    """ResNet-18 up to layer3: (B, in_ch, H, W) -> (B, 256, H/16, W/16).
    For in_ch != 3 the first conv's ImageNet filters are averaged and tiled."""
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if imagenet else None)
    if in_ch != 3:
        old = m.conv1
        m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            m.conv1.weight.copy_(old.weight.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * 3 / in_ch)
    return nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3)


# ========================================================== fusion bottleneck
class FusionBottleneck(nn.Module):
    """Concatenate both streams; the hidden one is replaced by a learned token."""

    def __init__(self, ch=FEAT):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.mix = nn.Sequential(
            nn.Conv2d(2 * ch, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))

    def forward(self, f_evt, f_dense, hide=None):            # hide: None|"event"|"dense"
        if hide == "event":
            f_evt = self.mask_token.expand_as(f_evt)
        elif hide == "dense":
            f_dense = self.mask_token.expand_as(f_dense)
        return self.mix(torch.cat([f_evt, f_dense], 1))


# ===================================================================== decoder
def up_block(i, o):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                         nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


def decoder(out_ch, ch=FEAT):
    """(B, 256, H/16, W/16) -> (B, out_ch, H, W). Deliberately small, so the
    encoders (not the decoder) have to do the work."""
    return nn.Sequential(up_block(ch, 128), up_block(128, 64), up_block(64, 32), up_block(32, 32),
                         nn.Conv2d(32, out_ch, 3, padding=1))


# ================================================================= full model
class CrossModalAE(nn.Module):
    def __init__(self, dense_ch=DENSE_CH, event_bins=EVENT_BINS, imagenet=True):
        super().__init__()
        self.enc_event = resnet18_encoder(event_bins, imagenet)
        self.enc_dense = resnet18_encoder(dense_ch, imagenet)
        self.fuse = FusionBottleneck()
        self.dec_event = decoder(event_bins)
        self.dec_dense = decoder(dense_ch)

    def encode(self, ev, dn, hide=None):
        return self.fuse(self.enc_event(ev), self.enc_dense(dn), hide)

    def forward(self, ev, dn, hide):
        z = self.encode(ev, dn, hide)
        return self.dec_event(z) if hide == "event" else self.dec_dense(z)


# ============================================================ mask controller
def choose_hidden(direction):
    """direction: 'e2d' predict dense from events, 'd2e' the reverse, 'random'."""
    if direction == "e2d":
        return "dense"
    if direction == "d2e":
        return "event"
    return random.choice(["dense", "event"])


# ================================================================ weight maps
def event_weight_map(ev, mode="none", floor=0.1, blur=9, region=63):
    """Per-pixel loss weight (B,1,H,W) from the event voxel grid, mean = 1 so
    loss scale is comparable across variants.
      none     all ones                         -> ablation (b)
      count    |events| per pixel, smoothed     -> can be ~0 on static objects
      floor    count, but never below `floor`
      recency  later time bins count more
      ego      count minus a local per-region baseline: cheap ego-motion proxy.
               Background edges sweeping past fire everywhere in a region;
               independently moving objects stand out above that baseline.
               A stronger version: estimate global flow by contrast
               maximisation (Gallego et al., CVPR 2018) and keep residuals."""
    B, T, H, W = ev.shape
    if mode == "none":
        return torch.ones(B, 1, H, W, device=ev.device)
    a = ev.abs()
    if mode == "recency":
        a = a * torch.linspace(0.2, 1.0, T, device=ev.device).view(1, T, 1, 1)
    act = F.avg_pool2d(a.sum(1, keepdim=True), blur, 1, blur // 2)
    if mode == "ego":
        base = F.avg_pool2d(act, region, 1, region // 2, count_include_pad=False)
        act = F.relu(act - base)
    act = act / (act.amax(dim=(2, 3), keepdim=True) + 1e-6)
    if mode == "floor":
        act = act.clamp(min=floor)
    return act / (act.mean(dim=(2, 3), keepdim=True) + 1e-6)


def weighted_mse(pred, target, w):
    return (w * (pred - target).pow(2)).mean()


def pretrain_step(model, opt, ev, dn, direction="e2d", weight_mode="ego"):
    hide = choose_hidden(direction)
    pred = model(ev, dn, hide)
    target = dn if hide == "dense" else ev
    w = event_weight_map(ev, weight_mode)     # computed from events, used ONLY in the loss
    loss = weighted_mse(pred, target, w)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


# ============================================================ stage 2: detect
class FusedBackbone(nn.Module):
    """Pretrained encoders + fusion, wrapped for torchvision FCOS.
    FCOS expects one image tensor, so event and dense channels are stacked:
    x = cat([event_voxel_grid, dense_frame], dim=1)."""

    def __init__(self, ae, event_bins=EVENT_BINS, freeze=True, train_fusion=False):
        super().__init__()
        self.enc_event, self.enc_dense, self.fuse = ae.enc_event, ae.enc_dense, ae.fuse
        self.event_bins, self.freeze, self.train_fusion = event_bins, freeze, train_fusion
        self.out_channels = FEAT
        # trainable extra levels. p3 (stride 8) exists so that small objects
        # (pedestrians, distant cars) contain at least one prediction point.
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 nn.Conv2d(FEAT, FEAT, 3, padding=1))
        self.down5 = nn.Conv2d(FEAT, FEAT, 3, 2, 1)
        self.down6 = nn.Conv2d(FEAT, FEAT, 3, 2, 1)
        if freeze:
            for m in self._pretrained():
                m.requires_grad_(False)

    def _pretrained(self):
        if self.train_fusion:        # encoders frozen, fusion layers trained with the head
            return (self.enc_event, self.enc_dense)
        return (self.enc_event, self.enc_dense, self.fuse)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:                     # keep BatchNorm statistics frozen too
            for m in self._pretrained():
                m.eval()
        return self

    def forward(self, x):
        ev, dn = x[:, :self.event_bins], x[:, self.event_bins:]
        p4 = self.fuse(self.enc_event(ev), self.enc_dense(dn))       # stride 16
        p3 = self.up3(p4)                                             # stride 8
        p5 = self.down5(F.relu(p4))                                   # stride 32
        p6 = self.down6(F.relu(p5))                                   # stride 64
        return OrderedDict([("p3", p3), ("p4", p4), ("p5", p5), ("p6", p6)])


def build_detector(ae, num_classes=4, dense_ch=DENSE_CH, event_bins=EVENT_BINS,
                   freeze=True, size=(512, 640), score_thresh=0.05, train_fusion=False):
    """num_classes includes background: vehicle, pedestrian, bicycle -> 4,
    with labels 1..3 in the targets. Inputs must already be normalised.
    Anchor size = stride on every level (torchvision's FCOS convention), so FCOS
    assigns objects up to ~64 px to p3, 64-128 px to p4, 128-256 to p5, larger to p6.
    score_thresh: torchvision's default of 0.2 discards most detections of a
    weakly trained head and makes mAP read ~0; 0.05 is the usual eval setting."""
    c = event_bins + dense_ch
    anchors = AnchorGenerator(sizes=((8,), (16,), (32,), (64,)), aspect_ratios=((1.0,),) * 4)
    return FCOS(FusedBackbone(ae, event_bins, freeze, train_fusion), num_classes=num_classes,
                anchor_generator=anchors, image_mean=[0.0] * c, image_std=[1.0] * c,
                min_size=min(size), max_size=max(size), score_thresh=score_thresh,
                detections_per_img=100)


class LinearProbe(nn.Module):
    """Even cheaper alternative to FCOS: frozen features -> 1x1 conv predicting a
    per-cell class heatmap (CenterNet-style). Good first sanity check."""

    def __init__(self, ae, num_classes=3, event_bins=EVENT_BINS):
        super().__init__()
        self.backbone = FusedBackbone(ae, event_bins, freeze=True)
        self.head = nn.Conv2d(FEAT, num_classes, 1)

    def forward(self, x):
        return self.head(self.backbone(x)["p4"])


# ================================================================ experiments
EXPERIMENTS = {
    # professor's three-way comparison
    "a_random_init": None,                                   # skip pretraining
    "b_plain":       dict(direction="random", weight_mode="none"),
    "c_count":       dict(direction="random", weight_mode="count"),
    "c_floor":       dict(direction="random", weight_mode="floor"),
    "c_recency":     dict(direction="random", weight_mode="recency"),
    "c_ego":         dict(direction="random", weight_mode="ego"),
    # 2x2: mask direction x weighting (reuse best weight variant for "weighted")
    "e2d_plain":     dict(direction="e2d", weight_mode="none"),
    "e2d_weighted":  dict(direction="e2d", weight_mode="ego"),
    "d2e_plain":     dict(direction="d2e", weight_mode="none"),
    "d2e_weighted":  dict(direction="d2e", weight_mode="ego"),
}
LABEL_FRACTIONS = [0.10, 0.25, 0.50, 1.00]   # subsample by SEQUENCE, not frame


if __name__ == "__main__":                    # smoke test with random tensors
    ae = CrossModalAE(imagenet=False)
    opt = torch.optim.AdamW(ae.parameters(), lr=1e-4)
    ev, dn = torch.randn(2, EVENT_BINS, 512, 640), torch.randn(2, DENSE_CH, 512, 640)
    for mode in ["none", "count", "floor", "recency", "ego"]:
        print(mode, pretrain_step(ae, opt, ev, dn, "random", mode))

    det = build_detector(ae)
    det.train()
    x = list(torch.cat([ev, dn], 1))
    tgt = [{"boxes": torch.tensor([[10., 20., 120., 140.]]), "labels": torch.tensor([1])}] * 2
    print({k: round(v.item(), 3) for k, v in det(x, tgt).items()})
    det.eval()
    print(det(x)[0]["boxes"].shape)