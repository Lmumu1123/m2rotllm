"""Adapt MBHM dataset so RotLLM pre_train / fine_tune can consume it.

MBHM differences vs RotLLM LMR:
  - ~135k bearing samples, labels 0-9 (RotLLM LMR has ~237k, labels 0-14 with gears)
  - HDF5 key is `vibration` (RotLLM expects `data`)
  - Same DCN length 24000 and similar sqlite metadata schema
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd


ROOT = os.path.dirname(os.path.abspath(__file__))
MBHM_DIR = os.environ.get(
    "MBHM_DIR",
    "/data1/datasets/RotLLM/mbhm_dataset"
    if os.path.isdir("/data1/datasets/RotLLM/mbhm_dataset")
    else os.path.join(ROOT, "mbhm_dataset"),
)


def parquet_to_sqlite(
    parquet_path: str | None = None,
    sqlite_path: str | None = None,
) -> str:
    parquet_path = parquet_path or os.path.join(MBHM_DIR, "metadata.parquet")
    sqlite_path = sqlite_path or os.path.join(MBHM_DIR, "metadata.sqlite")
    df = pd.read_parquet(parquet_path)

    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE condition (
            condition_id INTEGER PRIMARY KEY,
            dataset TEXT,
            component TEXT,
            code TEXT,
            channel INTEGER,
            rpm TEXT,
            load TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE file_info (
            file_id INTEGER PRIMARY KEY,
            condition_id INTEGER,
            label INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE label_note (
            label INTEGER PRIMARY KEY,
            note TEXT
        )
        """
    )

    cond = (
        df.groupby("condition_id", as_index=False)
        .agg(
            dataset=("dataset", "first"),
            code=("bearing_code", "first"),
            channel=("channel", "first"),
            rpm=("rpm", "first"),
            load=("load", "first"),
        )
        .sort_values("condition_id")
    )
    cond["component"] = "Bearing"
    cur.executemany(
        "INSERT INTO condition VALUES (?,?,?,?,?,?,?)",
        [
            (
                int(r.condition_id),
                r.dataset,
                r.component,
                r.code,
                int(r.channel),
                str(r.rpm),
                str(r.load),
            )
            for r in cond.itertuples(index=False)
        ],
    )

    cur.executemany(
        "INSERT INTO file_info VALUES (?,?,?)",
        [
            (int(r.file_id), int(r.condition_id), int(r.label))
            for r in df[["file_id", "condition_id", "label"]].itertuples(index=False)
        ],
    )

    label_notes = {
        0: "Fault-Free",
        1: "Minor Inner Ring Fault",
        2: "Moderate Inner Ring Fault",
        3: "Severe Inner Ring Fault",
        4: "Minor Ball Fault",
        5: "Moderate Ball Fault",
        6: "Severe Ball Fault",
        7: "Minor Outer Ring Fault",
        8: "Moderate Outer Ring Fault",
        9: "Severe Outer Ring Fault",
    }
    cur.executemany(
        "INSERT INTO label_note VALUES (?,?)",
        list(label_notes.items()),
    )
    conn.commit()
    conn.close()
    print(f"wrote {sqlite_path} files={len(df)} conditions={len(cond)}")
    return sqlite_path


def link_vibration_as_data(src_hdf5: str | None = None, dst_hdf5: str | None = None) -> str:
    """Create an HDF5 with dataset name `data` pointing at MBHM vibration array.

    If creating a full copy is too expensive, write a thin wrapper HDF5 that
    externally links to `vibration` as `data`.
    """
    import h5py

    src_hdf5 = src_hdf5 or os.path.join(MBHM_DIR, "data.hdf5")
    dst_hdf5 = dst_hdf5 or os.path.join(MBHM_DIR, "data_as_rotllm.hdf5")
    if not os.path.exists(src_hdf5):
        raise FileNotFoundError(src_hdf5)
    if os.path.exists(dst_hdf5):
        os.remove(dst_hdf5)
    with h5py.File(dst_hdf5, "w") as f:
        f["data"] = h5py.ExternalLink(os.path.abspath(src_hdf5), "/vibration")
    print(f"wrote wrapper {dst_hdf5} -> {src_hdf5}:/vibration")
    return dst_hdf5


if __name__ == "__main__":
    parquet_to_sqlite()
    hdf5 = os.path.join(MBHM_DIR, "data.hdf5")
    if os.path.exists(hdf5):
        link_vibration_as_data()
    else:
        print(f"skip hdf5 wrapper; missing {hdf5}")
