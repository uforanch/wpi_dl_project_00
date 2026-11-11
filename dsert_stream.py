"""
Stream DSERT-RoLL from Hugging Face frame by frame, keeping all sensors of a
frame together in one sample. Nothing is downloaded up front: each sequence's
label.pkl is fetched first, then each frame's sensor files are fetched together
(in parallel, with a few frames prefetched ahead).

    pip install -U huggingface_hub numpy pillow torch
    huggingface-cli login   # optional, but raises Hugging Face rate limits

Usage:
    ds = DsertStream(split="train", sensors=["rgb_l", "livox", "radar"], max_sequences=2)
    for sample in ds:
        sample["sensors"]["rgb_l"]   # HxWx3 uint8
        sample["sensors"]["livox"]   # Nx? float points
        sample["annos"]              # 3D boxes for this frame
        break

    # With PyTorch workers (sequences are spread across workers):
    loader = DataLoader(ds, batch_size=None, num_workers=4)
"""
import io
import pickle
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from huggingface_hub import HfFileSystem
from torch.utils.data import IterableDataset, get_worker_info

REPO = "datasets/jeongyh98/DSERT-RoLL"

# short name -> key in label.pkl's info[i]["sensor"]
SENSORS = {
    "rgb_l": "rgb_left_path",
    "rgb_r": "rgb_right_path",
    "event_l": "event_left_path",
    "event_r": "event_right_path",
    "thermal_l": "thermal_left_path",
    "thermal_r": "thermal_right_path",
    "livox": "livox_path",
    "ouster": "ouster_path",
    "radar": "radar_path",
}


def _decode(path, data):
    p = path.lower()
    if p.endswith((".jpg", ".jpeg", ".png")):
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    if p.endswith(".npy"):
        return np.load(io.BytesIO(data), allow_pickle=False)
    if p.endswith(".npz"):
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    return data  # unknown type: raw bytes


class DsertStream(IterableDataset):
    def __init__(
        self,
        split="train",
        sensors=tuple(SENSORS),
        sequences=None,          # explicit list like ["Fog/2025_04_15_07_10_51"]
        max_sequences=None,      # cap for quick tests
        shuffle_sequences=False,
        seed=0,
        revision="main",
        threads=16,              # parallel file fetches per worker
        prefetch_frames=4,       # frames fetched ahead of the one being yielded
        decode=True,
        retries=3,
    ):
        unknown = set(sensors) - set(SENSORS)
        if unknown:
            raise ValueError(f"unknown sensors {unknown}; choose from {list(SENSORS)}")
        self.split, self.sensors, self.sequences = split, list(sensors), sequences
        self.max_sequences, self.shuffle, self.seed = max_sequences, shuffle_sequences, seed
        self.revision, self.threads, self.prefetch = revision, threads, prefetch_frames
        self.decode, self.retries = decode, retries

    # ---- helpers -------------------------------------------------------
    def _fs(self):
        return HfFileSystem()

    def _root(self):
        return f"{REPO}@{self.revision}" if self.revision != "main" else REPO

    def _read(self, fs, path):
        for attempt in range(self.retries):
            try:
                return fs.cat_file(path)
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def _sequence_list(self, fs):
        if self.sequences is not None:
            seqs = list(self.sequences)
        else:
            text = self._read(fs, f"{self._root()}/split/{self.split}.txt").decode()
            seqs = [l.strip().strip("/") for l in text.splitlines() if l.strip()]
        if self.shuffle:
            random.Random(self.seed).shuffle(seqs)
        if self.max_sequences:
            seqs = seqs[: self.max_sequences]
        return seqs

    def _resolve(self, seq_rel, p):
        """label.pkl paths are 'relative'; accept seq-relative, weather-relative or root-relative."""
        p = str(p).replace("\\", "/").lstrip("./")
        weather, seq_name = seq_rel.split("/", 1)
        if p.startswith(seq_rel + "/"):
            rel = p
        elif p.startswith(seq_name + "/"):
            rel = f"{weather}/{p}"
        elif "/" + seq_rel + "/" in "/" + p:  # absolute path from the authors' machine
            rel = p[p.index(seq_rel):]
        else:
            rel = f"{seq_rel}/{p}"
        return f"{self._root()}/{rel}"

    def _fetch_frame(self, fs, pool, seq_rel, frame):
        paths = {s: self._resolve(seq_rel, frame["sensor"][SENSORS[s]]) for s in self.sensors
                 if frame["sensor"].get(SENSORS[s])}
        futs = {s: pool.submit(self._read, fs, p) for s, p in paths.items()}
        return paths, futs

    # ---- iteration -----------------------------------------------------
    def __iter__(self):
        fs = self._fs()
        seqs = self._sequence_list(fs)
        wi = get_worker_info()
        if wi is not None:  # give each DataLoader worker its own sequences
            seqs = seqs[wi.id :: wi.num_workers]

        with ThreadPoolExecutor(self.threads) as pool:
            for seq_rel in seqs:
                # label.pkl is the dataset's official annotation format (pickle).
                label = pickle.loads(self._read(fs, f"{self._root()}/{seq_rel}/label.pkl"))
                meta, frames = label["meta"], label["info"]

                pending = deque()
                it = iter(frames)
                for frame in it:  # fill the prefetch window
                    pending.append((frame, *self._fetch_frame(fs, pool, seq_rel, frame)))
                    if len(pending) >= self.prefetch:
                        break

                while pending:
                    frame, paths, futs = pending.popleft()
                    nxt = next(it, None)
                    if nxt is not None:
                        pending.append((nxt, *self._fetch_frame(fs, pool, seq_rel, nxt)))

                    data = {s: f.result() for s, f in futs.items()}
                    if self.decode:
                        data = {s: _decode(paths[s], b) for s, b in data.items()}

                    yield {
                        "sequence": seq_rel,
                        "frame_idx": frame.get("frame_idx", frame.get("sample_idx")),
                        "timestamp": frame.get("time_stamp"),
                        "pose": frame.get("pose"),
                        "annos": frame.get("annos"),
                        "weather": meta.get("weather"),
                        "light": meta.get("light"),
                        "calibration": meta.get("calibration"),
                        "sensors": data,
                    }


if __name__ == "__main__":
    ds = DsertStream(split="val", sensors=["rgb_l", "thermal_l", "livox", "radar"], max_sequences=1, shuffle_sequences=True)
    for i, s in enumerate(ds):
        shapes = {k: (v.shape if hasattr(v, "shape") else {kk: vv.shape for kk, vv in v.items()})
                  for k, v in s["sensors"].items()}
        print(s["sequence"], s["frame_idx"], shapes)
        if i == 4:
            break