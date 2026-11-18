"""
The model: a two-stream cross-modal masked autoencoder (event + thermal/RGB).

    event voxel grid -> event encoder --+
                                         +-> fusion bottleneck -> decoder -> HIDDEN modality
    dense frame ----> dense encoder ----+    (hidden stream replaced by a learned mask token)

    loss = event-derived weight map x reconstruction error

Everything else (data, experiment list, detector, evaluation) lives in
run_experiments.py. To try a different model, see "MODEL" at the top of
run_experiments.py; any replacement must follow the same small interface as
CrossModalAE below (forward, encode, out_channels, fusion_modules).

Input sizes: H and W must be divisible by 16 (encoder stride).
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

EVENT_BINS = 5      # same as DSERT-RoLL
DENSE_CH = 1        # 1 = thermal, 3 = RGB
FEAT = 256          # ResNet-18 layer3 width


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
    """The interface run_experiments.py relies on (any replacement model needs it):
        forward(ev, dn, hide) -> reconstruction of the hidden modality   [pretrain]
        encode(ev, dn)        -> fused features (B, out_channels, H/16, W/16) [probe]
        out_channels          -> channels of encode()'s output
        fusion_modules()      -> modules that `probe --train-fusion` trains
    """
    out_channels = FEAT

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

    def fusion_modules(self):
        return [self.fuse]


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


# ============================================================== one train step
def pretrain_step(model, opt, ev, dn, direction="random", weight_mode="none"):
    """One pretraining step. `direction` and `weight_mode` come from an entry
    of EXPERIMENTS in run_experiments.py."""
    hide = choose_hidden(direction)
    pred = model(ev, dn, hide)
    target = dn if hide == "dense" else ev
    w = event_weight_map(ev, weight_mode)     # computed from events, used ONLY in the loss
    loss = weighted_mse(pred, target, w)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


if __name__ == "__main__":      # quick shape check: python event_dense_ae.py
    m = CrossModalAE(imagenet=False)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    ev, dn = torch.randn(2, EVENT_BINS, 256, 384), torch.randn(2, DENSE_CH, 256, 384)
    print("encode ->", tuple(m.encode(ev, dn).shape))
    for mode in ("none", "count", "floor", "recency", "ego"):
        print(mode, round(pretrain_step(m, opt, ev, dn, "random", mode), 4))