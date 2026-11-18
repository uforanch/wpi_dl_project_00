"""
VideoMAE version of the cross-modal masked autoencoder.

Same design as CrossModalAE in event_dense_ae.py, with ONE change: each
modality's encoder is a pretrained VideoMAE vision transformer (Hugging Face
`transformers`) instead of a ResNet-18. Fusion bottleneck, mask token, decoders,
loss weighting and the pretraining step are reused unchanged from
event_dense_ae.py, so CrossModalAE vs. VideoMAEAE compares the encoders only.

    event voxel grid -> 1x1 conv (5 -> 3 ch) -> VideoMAE encoder -> 1x1 proj --+
                                                                                +-> fusion -> decoder
    dense frame -----> repeat to 3 ch -------> VideoMAE encoder -> 1x1 proj --+

How a single frame is fed to a video model: each frame is repeated to make a
2-frame clip (VideoMAE's tubelet spans 2 frames), so every 16x16 patch becomes
one token. Output tokens are reshaped back into a (H/16, W/16) feature map,
the same shape the ResNet encoders produce.

Use it with:  python run_experiments.py <command> --model videomae ...
Needs:        pip install "transformers>=4.30"   (first use downloads ~350 MB)

Input size must equal the --size given to run_experiments.py (fixed at build
time because VideoMAE checks it), and both sides must be divisible by 16.
"""
import torch
import torch.nn as nn

from event_dense_ae import FEAT, FusionBottleneck, decoder

# Self-supervised VideoMAE ViT-Base, pretrained on Kinetics-400 by masked video
# reconstruction (no labels). Any VideoMAE checkpoint on the Hub works here.
VIDEOMAE_CHECKPOINT = "MCG-NJU/videomae-base"


def videomae_encoder(size, pretrained=True, checkpoint=VIDEOMAE_CHECKPOINT):
    """A VideoMAE encoder configured for single frames of `size` = (H, W).
    The fixed sinusoidal position table is rebuilt for this size, so the
    pretrained weights load unchanged."""
    from transformers import VideoMAEConfig, VideoMAEModel
    overrides = dict(image_size=tuple(size), num_frames=2)
    if pretrained:
        return VideoMAEModel.from_pretrained(checkpoint, **overrides)
    return VideoMAEModel(VideoMAEConfig.from_pretrained(checkpoint, **overrides))


class VideoMAEAE(nn.Module):
    """Same interface as event_dense_ae.CrossModalAE:
        forward(ev, dn, hide) -> reconstruction of the hidden modality   [pretrain]
        encode(ev, dn)        -> fused features (B, 256, H/16, W/16)     [probe]
        out_channels, fusion_modules()
    `imagenet=True` here means "start from the pretrained VideoMAE weights"
    (Kinetics video, not ImageNet); the flag name is kept so run_experiments.py
    treats both models the same way."""
    out_channels = FEAT

    def __init__(self, dense_ch, event_bins, imagenet=True, size=(256, 384),
                 checkpoint=VIDEOMAE_CHECKPOINT):
        super().__init__()
        if size[0] % 16 or size[1] % 16:
            raise ValueError(f"VideoMAE needs H and W divisible by 16, got {size}")
        self.size = tuple(size)
        self.enc_event = videomae_encoder(size, imagenet, checkpoint)
        self.enc_dense = videomae_encoder(size, imagenet, checkpoint)
        D = self.enc_event.config.hidden_size                  # 768 for ViT-Base
        # events (5 time bins) -> 3 channels; starts as the plain event count
        self.event_in = nn.Conv2d(event_bins, 3, 1, bias=False)
        nn.init.constant_(self.event_in.weight, 1.0 / event_bins)
        self.dense_ch = dense_ch
        # transformer width -> the 256 channels the fusion block and decoders expect
        self.proj_event = nn.Sequential(nn.Conv2d(D, FEAT, 1, bias=False), nn.BatchNorm2d(FEAT),
                                        nn.ReLU(inplace=True))
        self.proj_dense = nn.Sequential(nn.Conv2d(D, FEAT, 1, bias=False), nn.BatchNorm2d(FEAT),
                                        nn.ReLU(inplace=True))
        self.fuse = FusionBottleneck()
        self.dec_event = decoder(event_bins)
        self.dec_dense = decoder(dense_ch)

    # ---------------------------------------------------------------- helpers
    def _to_rgb(self, x, is_event):
        if is_event:
            return self.event_in(x)
        if x.shape[1] == 3:
            return x
        return x[:, :1].expand(-1, 3, -1, -1)              # 1-channel thermal -> 3

    def _tokens_to_map(self, enc, x):
        """(B,3,H,W) -> encoder -> (B, D, H/16, W/16)."""
        clip = x.unsqueeze(1).expand(-1, 2, -1, -1, -1)     # (B, 2 frames, 3, H, W)
        tok = enc(pixel_values=clip.contiguous()).last_hidden_state   # (B, N, D)
        B, N, D = tok.shape
        h, w = x.shape[2] // 16, x.shape[3] // 16
        return tok.transpose(1, 2).reshape(B, D, h, w)

    def _features(self, x, enc, proj, is_event):
        return proj(self._tokens_to_map(enc, self._to_rgb(x, is_event)))

    # -------------------------------------------------------------- interface
    def encode(self, ev, dn, hide=None):
        B, _, H, W = ev.shape
        shape = (B, FEAT, H // 16, W // 16)
        # the hidden modality's features are replaced by the mask token anyway,
        # so its (expensive) transformer is skipped instead of run and discarded
        f_evt = ev.new_zeros(shape) if hide == "event" else \
            self._features(ev, self.enc_event, self.proj_event, True)
        f_dn = dn.new_zeros(shape) if hide == "dense" else \
            self._features(dn, self.enc_dense, self.proj_dense, False)
        return self.fuse(f_evt, f_dn, hide)

    def forward(self, ev, dn, hide):
        z = self.encode(ev, dn, hide)
        return self.dec_event(z) if hide == "event" else self.dec_dense(z)

    def fusion_modules(self):
        """Not pretrained by VideoMAE itself, so `probe --train-fusion` trains them."""
        return [self.event_in, self.proj_event, self.proj_dense, self.fuse]


if __name__ == "__main__":      # quick shape check: python videomae_ae.py
    from event_dense_ae import pretrain_step
    m = VideoMAEAE(dense_ch=1, event_bins=5, imagenet=False, size=(256, 384))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    ev, dn = torch.randn(2, 5, 256, 384), torch.randn(2, 1, 256, 384)
    print("encode ->", tuple(m.encode(ev, dn).shape))
    print("pretrain step loss", round(pretrain_step(m, opt, ev, dn, "random", "ego"), 4))
    print(f"{sum(p.numel() for p in m.parameters()) / 1e6:.0f} M parameters")
