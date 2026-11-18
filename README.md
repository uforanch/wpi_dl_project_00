# Event + thermal cross-modal masked autoencoder on DSERT-RoLL

Self-supervised pretraining for a two-stream (event camera + thermal or RGB)
autoencoder, evaluated by how well the pretrained encoders support 2D object
detection when only a fraction of the labels is available.

The core idea: hide one modality entirely, make the network reconstruct it from
the other, and weight the reconstruction loss by **where the event camera
actually fired**, optionally after removing activity caused by the car's own
motion (ego-motion).

---

## Files

| File | What it is |
|---|---|
| `run_experiments.py` | Everything except the model: data download/caching, the experiment list, the detector used for evaluation, metrics, plots |
| `event_dense_ae.py` | The model only: encoders, fusion bottleneck, decoders, loss weight maps, one pretraining step |
| `dsert_stream.py` | Reads DSERT-RoLL files from Hugging Face (used by `cache`) |
| `pyproject.toml` | Dependencies, for `pip install -e .` or `uv sync` |
| `run_experiments.sbatch` | Example Slurm job for a GPU cluster |

All files go in the same folder.

## Setup

Python 3.8+ (3.10+ recommended).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # or: uv sync   (environment goes in ./.venv)
```

On a GPU machine with pip, install the CUDA build of PyTorch **before**
`pip install -e .` (command from pytorch.org). Installing also adds a
`dsert-exp` command (`dsert-exp inspect` = `python run_experiments.py inspect`).

**Hugging Face token** (optional, raises download rate limits). Either set the
`HF_TOKEN` environment variable, which `huggingface_hub` reads automatically,
or pass `--hftoken <token>`. In Colab:

```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HFTOKEN")    # name of your Colab secret
```

---

## Quick start

```bash
python run_experiments.py cache --peek                # 1. check data and label formats
python run_experiments.py cache --all-sequences --max-gb 5   # 2. download + cache (resumable)
python run_experiments.py inspect                     # 3. shapes, labels, alignment images
python run_experiments.py overlap                     # 4. heatmap-vs-box check (no training)
python run_experiments.py pretrain                    # 5. pretrain every experiment
python run_experiments.py probe --train-fusion        # 6. detector at 10/25/50/100 % labels
python run_experiments.py plot                        # 7. label-efficiency curve + 2x2 table
```

Before a full run, test the whole chain quickly with `--steps 200` on
`pretrain` and `probe`.

---

## What the pipeline does

### Stage 1: label-free pretraining (`pretrain`)

```
event voxel grid ──► event encoder ──┐
                                      ├─► fusion bottleneck ──► decoder ──► reconstruct the HIDDEN modality
thermal frame ────► thermal encoder ─┘    (hidden stream replaced
                                            by a learned mask token)

loss = weight map × reconstruction error      (weight map computed from event activity)
```

- **Encoders:** torchvision ResNet-18 up to `layer3` (256 channels, 1/16
  resolution), starting from ImageNet weights. The event encoder's first layer
  is widened to 5 input channels.
- **Fusion bottleneck:** the hidden modality's features are replaced with a
  learned mask token; both streams are concatenated and mixed by a 1×1 and a
  3×3 convolution.
- **Decoders:** one per modality, four upsample + conv blocks each. Small on
  purpose, so the encoders do the real work.
- **Everything is trained** (encoders, fusion, mask token, decoders), starting
  fresh for each experiment. In a given step only the parts that were used
  receive gradients: the hidden modality's encoder and the visible modality's
  decoder don't.
- **Mask direction:** `e2d` (predict thermal from events), `d2e` (the reverse),
  or `random` (either, chosen per step).
- **Weight map modes:**

  | Mode | Weight per pixel |
  |---|---|
  | `none` | uniform (plain, unweighted loss) |
  | `count` | number of events, smoothed |
  | `floor` | `count`, but never below a minimum, so static objects still count |
  | `recency` | `count`, with later events weighted more |
  | `ego` | `count` minus each region's average activity, a rough proxy for removing ego-motion |

Each experiment's model is saved to `<out>/ckpt/<name>.pt` (all weights,
including the decoders, plus its settings). These are **autoencoders, not
detectors**.

### Stage 2: evaluation (`probe`)

For every experiment and every label fraction, a **brand-new detector** is
built on top of that experiment's pretrained model, trained from scratch for a
fixed number of steps, scored on the validation set, and discarded. Only the
scores are kept (`<out>/probe.csv`). Every run uses the same detector design and
budget, so differences come from the pretrained features.

| Part | During `probe` |
|---|---|
| Event and thermal encoders | frozen (trained only with `--finetune`) |
| Fusion layers | frozen by default; **trained with `--train-fusion`** |
| Extra feature levels (strides 8, 32, 64) | trained from scratch |
| FCOS detection head | trained from scratch (`--head-convs` sets its size) |
| Decoders | not used |

Scores are 2D AP at IoU 0.5 per class (Bike, Pedestrian, Vehicle) and their
mean. Vehicle AP is the most reliable number, and it is the class the
DSERT-RoLL paper reports for 2D detection.

### Experiments

Defined in `EXPERIMENTS` near the top of `run_experiments.py`:

| Name | Mask direction | Weighting | Purpose |
|---|---|---|---|
| `a_random_init` | n/a | n/a | baseline: no pretraining, random weights |
| `a_imagenet` | n/a | n/a | baseline: ImageNet weights, no pretraining (added by `probe`) |
| `b_plain` | random | none | pretraining with an unweighted loss |
| `c_count`, `c_floor`, `c_recency`, `c_ego` | random | each mode | effect of each weighting signal |
| `e2d_plain`, `e2d_weighted` | events → thermal | none / ego | 2×2 of direction × weighting |
| `d2e_plain`, `d2e_weighted` | thermal → events | none / ego | 2×2 of direction × weighting |

A full `probe` is 11 models × 4 label fractions = 44 detector trainings.

---

## Commands and options

### `cache`: download and cache data

Uses the official train/val split from Hugging Face. Validation sequences are
cached first, then training sequences; both alternate between weather
conditions, so a size cap cuts every condition evenly. Re-running skips
finished sequences.

| Option | Default | Meaning |
|---|---|---|
| `--peek` | | print label, calibration and sensor formats for one sequence, then stop |
| `--all-sequences` | | cache the whole split |
| `--per-weather TRAIN VAL` | `4 2` | otherwise, how many sequences per weather condition |
| `--max-gb` | none | stop starting new sequences once the cache folder reaches this size (can overshoot by one sequence, ~30–40 MB) |
| `--frame-step` | `5` | keep every Nth frame |
| `--frames-per-seq` | `100` | maximum frames per sequence |
| `--cache-size H W` | `256 384` | **image resolution** frames are stored at (not a storage limit). About 1.1 MB per frame at the default; doubling both sides costs ~4× |
| `--box-camera` | `event_l` | whose 2D boxes to keep |
| `--no-align` | | resize thermal instead of warping it into the event camera's view |
| `--dense` | `thermal` | `thermal` or `rgb` |
| `--threads` | `16` | parallel downloads |

Output in `--cache` (default `dsert_frames_cache/`):
`<Weather>__<sequence>__event_l__<config>.npy` (event voxel grids
`(N, 5, H, W)`), `..._thermal_l_...npy` (aligned thermal `(N, H, W)`),
`meta/`, `labels/`, `raw_labels/` and `splits.json`. `<config>` is a code
derived from the cache settings. **Keep only one config in the folder**, or
sequences are counted twice.

Downloads are dominated by event files: roughly 5–7 MB per frame in clear and
rain scenes, much less in fog and snow.

### `labels`

Rebuilds `labels/*.json` from the saved `label.pkl` files without downloading.
Use after editing `extract_boxes_2d()`.

### `inspect`

Prints shapes, value ranges, the split and label counts, and saves images to
`<out>/inspect/`. The third row overlays events (red) on thermal; red edges
should sit on thermal object edges.

### `overlap`

Measures, on validation frames, how much of each weight map falls inside
ground-truth boxes: **mass in boxes**, **box area**, and **lift** (mass ÷ area;
1.0 = no better than uniform). Writes `<out>/overlap.csv` and figures to
`<out>/heatmaps/`. Needs no training.

### `pretrain`

Trains each experiment for `--steps` (default 5000) and saves
`<out>/ckpt/<name>.pt`. **Skips experiments that already have a checkpoint**:
use `--overwrite`, or a new `--out` folder, after changing the data.

| Option | Meaning |
|---|---|
| `--only NAME ...` | run only these experiments |
| `--overwrite` | retrain even if a checkpoint exists |
| `--no-imagenet` | start encoders from random weights instead of ImageNet |

Pretraining losses from different weight modes are not comparable with each
other (each weights pixels differently).

### `probe`

| Option | Default | Meaning |
|---|---|---|
| `--train-fusion` | off | **recommended.** Train the fusion layers with the head (encoders stay frozen). Without it, the baselines pass features through a random, frozen fusion layer; in testing the detector then barely learned (training mAP 0.04 vs. 0.28 with it) |
| `--finetune` | off | train the encoders too (not a probe any more; useful as an upper bound) |
| `--head-convs` | `4` | conv layers per FCOS head branch: `4` standard, `1` shallow (faster), `0` minimal (closest to a linear probe) |
| `--steps` | `3000` | training steps per detector, identical for every run |
| `--fractions F ...` | `0.1 0.25 0.5 1.0` | label fractions to run, e.g. `--fractions 0.5` |
| `--only NAME ...` | all | experiments to probe (baselines included by name) |
| `--score-thresh` | `0.05` | minimum detection confidence kept for scoring |
| `--max-val` | `1000` | maximum validation frames scored |

Each run prints validation **and** training-frame mAP, detections per frame and
mean top score. Training mAP near zero means the detector is not learning;
high training mAP with near-zero validation mAP means overfitting (too little
data).

Label subsets are blocks of consecutive frames, nested (the 10 % subset is
inside the 25 % one) and identical across runs with the same `--seed`.

**`probe.csv` does not record `--head-convs`, `--steps` or `--train-fusion`.**
Use a separate `--out` folder for each combination, or results get mixed.
Re-running `probe` appends rows rather than skipping finished runs; use
`--only` and `--fractions` to resume, and delete partial rows first.

### `plot`

Draws `<out>/label_efficiency.png` (mAP vs. label fraction for every run) and
prints the 2×2 direction × weighting table at the smallest label fraction.

### Options for every command

| Option | Default |
|---|---|
| `--cache` | `dsert_frames_cache` |
| `--out` | `results` |
| `--size H W` | `256 384` (match `--cache-size`) |
| `--dense` | `thermal` |
| `--device` | auto: cuda, then mps, then cpu |
| `--batch` | `8` |
| `--workers` | `2` (data-loading processes) |
| `--seed` | `0` |
| `--lr` | `3e-4` pretrain / `1e-3` probe |
| `--log-every` | `100` |
| `--hftoken` | none (or set `HF_TOKEN`) |

---

## Using a different model

`run_experiments.py` names the model in exactly one place, the **MODEL**
section at the top. Everything else (pretraining loop, checkpoints, detector,
evaluation) goes through three functions:

```python
def build_model(dense_ch, event_bins, imagenet=True):   # a fresh, untrained model
def train_step(model, opt, ev, dn, cfg):                 # one pretraining step -> loss
def weight_map(ev, mode):                                # used by `overlap`
```

The object `build_model` returns must provide:

| Member | Used by | Must return |
|---|---|---|
| `forward(ev, dn, hide)` | `pretrain` (via `train_step`) | reconstruction of the hidden modality (`hide` is `"event"` or `"dense"`) |
| `encode(ev, dn)` | `probe` | fused features, shape `(B, out_channels, H/16, W/16)` |
| `out_channels` | `probe` | number of channels `encode` returns |
| `fusion_modules()` | `probe --train-fusion` | list of sub-modules to train with the head (`[]` if none) |

`ev` is the event voxel grid `(B, event_bins, H, W)` and `dn` the dense frame
`(B, dense_ch, H, W)`. `encode` must output stride 16; for another stride,
change the `up3`/`down5`/`down6` levels and anchor sizes in the DETECTOR
section.

**Example:** a variant with a deeper fusion block. Put it in its own file,
`my_model.py`:

```python
import torch.nn as nn
from event_dense_ae import CrossModalAE, FusionBottleneck, FEAT

class DeepFusion(FusionBottleneck):
    def __init__(self, ch=FEAT):
        super().__init__(ch)
        self.mix = nn.Sequential(*self.mix, nn.Conv2d(ch, ch, 3, padding=1))   # one extra layer

class MyModel(CrossModalAE):
    def __init__(self, dense_ch, event_bins, imagenet=True):
        super().__init__(dense_ch, event_bins, imagenet)
        self.fuse = DeepFusion()
```

then change one line in `build_model`:

```python
def build_model(dense_ch, event_bins, imagenet=True):
    from my_model import MyModel
    return MyModel(dense_ch=dense_ch, event_bins=event_bins, imagenet=imagenet)
```

A model with a different training objective (another loss, a different masking
scheme) also replaces `train_step`. To add or remove runs, edit `EXPERIMENTS`:
each value is passed to `train_step` as `cfg`.

**Use a new `--out` folder** for a new model. Checkpoints are matched to
experiments by name only, and `probe` would otherwise load the old model's
weights into the new model.

Check a new model quickly with `python event_dense_ae.py` (shape check for the
default model), then `pretrain --steps 200 --only b_plain` and
`probe --steps 200 --only b_plain --fractions 1.0`.

---

## Running on Google Colab

Colab's `/content` folder is wiped whenever the runtime restarts, so keep the
cache and results on Drive and copy the cache to local disk each session:

```python
from google.colab import drive
drive.mount('/content/drive')
PROJ        = "/content/drive/MyDrive/Colab Notebooks"   # folder with the .py files
DRIVE_CACHE = "/content/drive/MyDrive/dsert_frames_cache"
RESULTS     = "/content/drive/MyDrive/dsert_results"
```
```
!python "$PROJ/run_experiments.py" cache --cache "$DRIVE_CACHE" --all-sequences --max-gb 5   # once
!rsync -a "$DRIVE_CACHE/" /content/dsert_frames_cache/                                       # each session
!cp "$PROJ"/*.py /content/
%cd /content
!python /content/run_experiments.py pretrain --out "$RESULTS"
!python /content/run_experiments.py probe --out "$RESULTS" --train-fusion
```

`$NAME` in `!` lines only works for **Python** variables. If one name can't be
filled in, Colab passes the whole line to the shell unchanged, and paths
like `"$PROJ/run_experiments.py"` silently become `/run_experiments.py`.

If you see `Transport endpoint is not connected`, the Drive connection dropped:
run `drive.flush_and_unmount()` and `drive.mount('/content/drive', force_remount=True)`,
or restart the session.

## Running on a Slurm cluster

`run_experiments.sbatch` requests one GPU and runs `inspect` → `overlap` →
`pretrain` → `probe --train-fusion` → `plot`. Before submitting, copy
`dsert_frames_cache/` to the cluster (compute nodes often have no internet),
and edit the environment lines and, if needed, the GPU partition. Then
`sbatch run_experiments.sbatch` and follow `slurm-<jobid>.out`.

---

## Known limitations

- **Small objects.** At 256×384 the median box is about 23×19 px and ~45 % are
  under 16 px on their shorter side. Detection at IoU 0.5 is hard at this
  resolution; a larger `--cache-size` helps at ~4× the storage and compute.
- **Data size.** With few sequences the detector overfits (high training mAP,
  near-zero validation mAP). Cache as many sequences as storage allows.
- **`ego` weighting is approximate.** It subtracts regional average activity
  and does not use the recorded vehicle pose.
- **Alignment assumes** each camera's `extrinsic` maps camera → RGB_L. The
  rotations are under half a degree, so the error is a few pixels at most;
  check the `inspect` overlay.
- **Fixed-direction runs leave one encoder untrained during pretraining.** In
  `e2d_*` the thermal encoder is always masked and keeps its ImageNet weights;
  likewise the event encoder in `d2e_*`.
- **One seed.** Results at small label fractions are noisy; repeat key runs
  with `--seed 1`, `--seed 2` before drawing conclusions.

## Data

DSERT-RoLL (CVPR 2026): stereo event, RGB and thermal cameras, 4D radar and
dual LiDAR, recorded in clear, fog, rain and snow. Hosted on Hugging Face at
`jeongyh98/DSERT-RoLL`. Check the dataset's licence before redistributing any
cached data.