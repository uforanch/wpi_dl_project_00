# Event + thermal cross-modal masked autoencoder on DSERT-RoLL

Self-supervised pretraining for a two-stream (event camera + thermal or RGB)
autoencoder, evaluated by how well the pretrained encoder supports 2D object
detection when only a fraction of the labels is available.

The core idea: hide one modality entirely, make the network reconstruct it from
the other, and weight the reconstruction loss by **where the event camera
actually fired**, optionally after removing activity caused by the car's own
motion (ego-motion).

---

## Files

| File | What it is |
|---|---|
| `run_experiments.py` | Command-line runner: download/cache data, run every experiment, score, plot |
| `event_dense_ae.py` | The model: encoders, fusion bottleneck, decoders, weight maps, detector wrapper, experiment list |
| `dsert_stream.py` | Streams DSERT-RoLL files from Hugging Face (used by the `cache` command) |
| `run_experiments.sbatch` | Example Slurm job for running everything on a GPU cluster |

All four files go in the same folder.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .             # installs dependencies from pyproject.toml
huggingface-cli login        # optional, raises Hugging Face rate limits
```

On a GPU machine, install the CUDA build of PyTorch **before** `pip install -e .`
(see pytorch.org for the right command for your CUDA version); pip will keep it.

Installing also adds a `dsert-exp` command, so `dsert-exp inspect` works the
same as `python run_experiments.py inspect`.

---

## Quick start

```bash
python run_experiments.py cache --peek     # 1. check data and label formats (downloads 1 frame)
python run_experiments.py cache            # 2. download + cache sequences (resumable)
python run_experiments.py inspect          # 3. sanity check: shapes, labels, alignment images
python run_experiments.py overlap          # 4. heatmap-vs-box measurement (no training)
python run_experiments.py pretrain         # 5. pretrain every experiment
python run_experiments.py probe            # 6. detector at 10/25/50/100 % of labels
python run_experiments.py plot             # 7. label-efficiency curve + 2x2 table
```

For a quick end-to-end test before a full run, add `--steps 200` to `pretrain`
and `probe`.

---

## What the pipeline does

### Stage 1: label-free pretraining

```
event voxel grid ──► event encoder ──┐
                                      ├─► fusion bottleneck ──► decoder ──► reconstruct the HIDDEN modality
thermal frame ────► thermal encoder ─┘    (hidden stream replaced
                                            by a learned mask token)

loss = weight map × reconstruction error      (weight map computed from event activity)
```

- **Encoders:** torchvision ResNet-18 up to `layer3` (256 channels, 1/16
  resolution), ImageNet-initialised. The event encoder's first conv is widened
  to 5 input channels.
- **Fusion bottleneck:** the hidden modality's features are replaced with a
  learned mask token, then both streams are concatenated and mixed with a 1×1
  and a 3×3 convolution.
- **Decoder:** four upsample + conv blocks back to full resolution.
- **Mask direction:** `e2d` (predict thermal from events), `d2e` (the reverse),
  or `random`.
- **Weight map modes:**

  | Mode | Weight per pixel |
  |---|---|
  | `none` | uniform (plain, unweighted loss) |
  | `count` | number of events, smoothed |
  | `floor` | `count`, but never below a minimum, so static objects still count |
  | `recency` | `count`, with later events weighted more |
  | `ego` | `count` minus each region's average activity, a rough proxy for removing ego-motion |

### Stage 2: evaluation

The pretrained encoders and fusion layers are **frozen**, and a torchvision
**FCOS** detection head is trained on top with 10 %, 25 %, 50 % and 100 % of
the labelled training frames. Every run gets the same number of training
steps. Results are 2D AP at IoU 0.5 per class.

### Experiments

Defined in `EXPERIMENTS` in `event_dense_ae.py`:

| Name | Mask direction | Weighting | Purpose |
|---|---|---|---|
| `a_random_init` | n/a | n/a | baseline: no pretraining, random weights |
| `a_imagenet` | n/a | n/a | baseline: ImageNet weights, no pretraining (added by `probe`) |
| `b_plain` | random | none | pretraining with an unweighted loss |
| `c_count`, `c_floor`, `c_recency`, `c_ego` | random | each mode | effect of each weighting signal |
| `e2d_plain`, `e2d_weighted` | events → thermal | none / ego | 2×2 of direction × weighting |
| `d2e_plain`, `d2e_weighted` | thermal → events | none / ego | 2×2 of direction × weighting |

---

## Commands in detail

### `cache`: download and cache data

Uses the official train/val split from Hugging Face. For each selected
sequence it saves:

- `dsert_frames_cache/<Weather>__<sequence>__event_l__<config>.npy`:
  event voxel grids `(N, 5, H, W)` of raw signed counts
- `dsert_frames_cache/<Weather>__<sequence>__thermal_l__<config>.npy`:
  thermal frames `(N, H, W)`, warped into the event camera's view
- `meta/`, `labels/`, `raw_labels/`: frame lists, 2D boxes in cached pixel
  coordinates, and each sequence's original `label.pkl`
- `splits.json`: which sequences are train and which are val

`<config>` is a short code derived from the cache settings, so caches made
with different settings don't overwrite each other. **Keep only one config in
the folder**, or every sequence will be counted twice.

Options:

| Option | Default | Meaning |
|---|---|---|
| `--peek` | | print label / calibration / sensor formats and stop |
| `--per-weather TRAIN VAL` | `4 2` | sequences per weather condition |
| `--all-sequences` | | cache the entire split instead |
| `--frame-step` | `5` | keep every Nth frame |
| `--frames-per-seq` | `100` | maximum frames per sequence |
| `--cache-size H W` | `256 384` | cached resolution |
| `--box-camera` | `event_l` | which camera's 2D boxes to keep |
| `--no-align` | | resize thermal instead of warping it to the event camera |
| `--dense` | `thermal` | `thermal` or `rgb` |

Re-running `cache` skips finished sequences. If only the alignment setting
changed, it re-downloads just the (small) thermal images.

Downloads are dominated by event files: roughly 5–7 MB per frame in clear and
rain scenes, much less in fog and snow.

### `labels`: rebuild 2D boxes offline

Re-reads the saved `label.pkl` files and rewrites `labels/*.json`. Use this
after editing `extract_boxes_2d()`; nothing is downloaded.

### `inspect`: sanity check

Prints array shapes, value ranges, the train/val split and label counts, and
saves images to `results/inspect/`. The third row of each image overlays
events (red) on thermal: red edges should sit on thermal object edges if the
alignment is right.

### `overlap`: heatmap-box measurement

For each weight mode, reports on validation frames:

- **mass in boxes**: share of the weight map that falls inside ground-truth
  boxes (the "energy-based pointing game")
- **box area**: share of the image covered by boxes
- **lift**: mass ÷ area. 1.0 means no better than a uniform map.

Writes `results/overlap.csv` and example figures to `results/heatmaps/`.

### `pretrain`

Trains each experiment for `--steps` (default 5000) and saves
`results/ckpt/<name>.pt`. Losses go to `results/pretrain_loss.csv`. Use
`--only NAME [NAME ...]` to run a subset, and `--overwrite` to retrain.

> Losses from different weight modes are **not comparable** with each other,
> since each mode weights pixels differently. Compare runs on the detection
> results instead.

### `probe`

For each checkpoint (plus both baselines) and each label fraction, trains the
FCOS head for `--steps` (default 3000) and appends AP@0.5 per class and mAP to
`results/probe.csv`. Label subsets are drawn as blocks of consecutive frames,
and each smaller subset is contained in the larger ones. `--finetune` unfreezes
the encoder instead of keeping it frozen.

### `plot`

Draws `results/label_efficiency.png` (mAP vs. label fraction for every run)
and prints the 2×2 direction × weighting table at the smallest label fraction.

### Common options

| Option | Default |
|---|---|
| `--cache` | `dsert_frames_cache` |
| `--out` | `results` |
| `--size H W` | `256 384` (should match `--cache-size`) |
| `--device` | auto: cuda, then mps, then cpu |
| `--batch` | `8` |
| `--workers` | `2` (data-loading processes) |
| `--seed` | `0` |
| `--lr` | `3e-4` pretrain / `1e-3` probe |

---

## Running on a Slurm cluster

`run_experiments.sbatch` requests one GPU and runs `inspect` → `overlap` →
`pretrain` → `probe` → `plot`. Before submitting:

1. Copy `dsert_frames_cache/` to the cluster, or run `cache` on the login node
   (compute nodes often have no internet).
2. Edit the environment lines (venv path, `module load`) and, if your cluster
   requires it, the GPU partition.
3. `sbatch run_experiments.sbatch`, then follow progress in `slurm-<jobid>.out`.

---

## Known limitations

- **Box format not yet confirmed on real data.** `extract_boxes_2d()` reads
  `annos['gt_boxes_event']` and accepts both 4-number boxes and projected
  corner points. Check the `cache --peek` output before trusting results.
- **`ego` weighting is approximate.** It subtracts regional average activity;
  it does not use the recorded vehicle pose. A pose-based version (removing
  rotation-induced events) would be closer to true ego-motion compensation.
- **Alignment assumes** each camera's `extrinsic` maps camera → RGB_L. The
  rotations involved are under half a degree, so the error is a few pixels at
  most either way; check the `inspect` overlay.
- **Fixed-direction runs leave one encoder untrained.** In `e2d_*` runs the
  thermal encoder is always masked, so it keeps its ImageNet weights;
  likewise the event encoder in `d2e_*`. Keep this in mind when reading the
  2×2 results.
- **Data size.** With default settings the cache holds on the order of a
  thousand frames. Results at 10 % of labels will be noisy; cache more
  sequences (`--per-weather`, `--all-sequences`) for a stronger curve.

## Data

DSERT-RoLL (CVPR 2026): stereo event, RGB and thermal cameras, 4D radar and
dual LiDAR, recorded in clear, fog, rain and snow. Hosted on Hugging Face at
`jeongyh98/DSERT-RoLL`. Check the dataset's licence before redistributing any
cached data.
