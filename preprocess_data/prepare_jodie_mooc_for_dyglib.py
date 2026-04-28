"""
Convert JODIE MOOC CSV to DyGLib processed_data format.

Input:
    JODIE data/mooc.csv

Output:
    processed_data/<dataset_name>/ml_<dataset_name>.csv
    processed_data/<dataset_name>/ml_<dataset_name>.npy
    processed_data/<dataset_name>/ml_<dataset_name>_node.npy

Recommended usage:
    python scripts/prepare_jodie_mooc_for_dyglib.py \
        --input /data/zyh/DeepProject/DyGLib/datasets/jodie/data/mooc.csv \
        --dyglib_root /data/zyh/DeepProject/DyGLib \
        --dataset_name mooc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_jodie_mooc(csv_path: Path) -> pd.DataFrame:
    """
    JODIE MOOC CSV has a compact header:
        user_id,item_id,timestamp,state_label,comma_separated_list_of_features

    But each data row actually has:
        user_id,item_id,timestamp,state_label,feat_0,feat_1,feat_2,feat_3

    Therefore, never read it by plain pd.read_csv(csv_path).
    """
    cols = [
        "user_id",
        "item_id",
        "timestamp",
        "state_label",
        "feat_0",
        "feat_1",
        "feat_2",
        "feat_3",
    ]

    df = pd.read_csv(csv_path, header=0, names=cols)

    # Type conversion
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["timestamp"] = df["timestamp"].astype(float)
    df["state_label"] = df["state_label"].astype(int)

    for col in ["feat_0", "feat_1", "feat_2", "feat_3"]:
        df[col] = df[col].astype(float)

    # Basic checks
    if not df["state_label"].isin([0, 1]).all():
        bad_values = sorted(df.loc[~df["state_label"].isin([0, 1]), "state_label"].unique())
        raise ValueError(f"state_label should be binary, but got values: {bad_values[:20]}")

    if not df["timestamp"].is_monotonic_increasing:
        print("[Warning] timestamp is not monotonic increasing. Sorting by timestamp stably.")
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    return df


def check_mooc_semantics(df: pd.DataFrame) -> None:
    """
    MOOC positive label should be a user-state-change action.
    Usually, positive actions are the final action of dropout users.
    Some users' final actions can still be label 0, because not every user drops out.
    """
    n = len(df)
    pos = int(df["state_label"].sum())
    neg = n - pos
    print("==== Raw JODIE MOOC statistics ====")
    print(f"#interactions: {n}")
    print(f"#negative labels: {neg}")
    print(f"#positive labels: {pos}")
    print(f"positive ratio: {pos / n:.8f}")
    print(f"#users: {df['user_id'].nunique()}")
    print(f"#items / targets: {df['item_id'].nunique()}")
    print(f"#edge feature dim: 4")

    # Check whether positive labels are on the last action of each user.
    last_action_indices = set(df.groupby("user_id", sort=False).tail(1).index.tolist())
    positive_indices = set(df.index[df["state_label"].eq(1)].tolist())
    violations = positive_indices - last_action_indices

    if len(violations) == 0:
        print("Positive-label check: all positive labels are on users' last actions.")
    else:
        print(
            "[Warning] Some positive labels are not on the last action of the corresponding user. "
            f"#violations = {len(violations)}"
        )


def build_dyglib_processed(
    df: pd.DataFrame,
    dyglib_root: Path,
    dataset_name: str,
    node_feat_dim: int = 172,
) -> None:
    """
    Build DyGLib processed files.

    DyGLib convention:
      - node ids start from 1; 0 is reserved for padding
      - edge ids start from 1; 0 is reserved for padding
      - for bipartite graphs, destination node ids are offset by #source nodes
    """
    output_dir = dyglib_root / "processed_data" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Factorize ids to be robust, even if raw ids are not perfectly consecutive.
    u_codes, u_uniques = pd.factorize(df["user_id"], sort=True)
    i_codes, i_uniques = pd.factorize(df["item_id"], sort=True)

    num_users = len(u_uniques)
    num_items = len(i_uniques)

    # DyGLib bipartite reindexing:
    # source: 1 ... num_users
    # target: num_users + 1 ... num_users + num_items
    src = u_codes.astype(np.int64) + 1
    dst = i_codes.astype(np.int64) + num_users + 1
    ts = df["timestamp"].to_numpy(dtype=np.float64)
    label = df["state_label"].to_numpy(dtype=np.float32)
    edge_idx = np.arange(1, len(df) + 1, dtype=np.int64)

    graph_df = pd.DataFrame(
        {
            "u": src,
            "i": dst,
            "ts": ts,
            "label": label,
            "idx": edge_idx,
        }
    )

    # Edge features: row 0 is padding, edge id 1 corresponds to first interaction.
    edge_feats_raw = df[["feat_0", "feat_1", "feat_2", "feat_3"]].to_numpy(dtype=np.float32)
    edge_feats = np.vstack(
        [
            np.zeros((1, edge_feats_raw.shape[1]), dtype=np.float32),
            edge_feats_raw,
        ]
    )

    # MOOC has no node features in DyGLib setting; use zeros.
    num_total_nodes = num_users + num_items
    node_feats = np.zeros((num_total_nodes + 1, node_feat_dim), dtype=np.float32)

    out_csv = output_dir / f"ml_{dataset_name}.csv"
    out_edge = output_dir / f"ml_{dataset_name}.npy"
    out_node = output_dir / f"ml_{dataset_name}_node.npy"
    out_meta = output_dir / f"ml_{dataset_name}_meta.json"

    graph_df.to_csv(out_csv, index=False)
    np.save(out_edge, edge_feats)
    np.save(out_node, node_feats)

    meta = {
        "dataset_name": dataset_name,
        "source": "JODIE MOOC CSV",
        "task": "dynamic node classification / user state change prediction",
        "label_definition": "state_label=1 iff the student drops out after this action",
        "num_interactions": int(len(df)),
        "num_users": int(num_users),
        "num_items": int(num_items),
        "num_total_nodes": int(num_total_nodes),
        "num_positive_labels": int(label.sum()),
        "positive_ratio": float(label.mean()),
        "edge_feat_dim_without_padding": int(edge_feats_raw.shape[1]),
        "edge_feat_shape_with_padding": list(edge_feats.shape),
        "node_feat_shape_with_padding": list(node_feats.shape),
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    print("\n==== Saved DyGLib processed files ====")
    print(out_csv)
    print(out_edge)
    print(out_node)
    print(out_meta)

    print("\n==== Processed file checks ====")
    print(graph_df.head())
    print("label distribution:")
    print(graph_df["label"].value_counts())
    print("edge_feats shape:", edge_feats.shape)
    print("node_feats shape:", node_feats.shape)


def print_dyglib_split_statistics(
    dyglib_root: Path,
    dataset_name: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> None:
    """
    DyGLib node classification uses chronological split by timestamp quantiles.
    Default is 70/15/15 when val_ratio=test_ratio=0.15.
    """
    path = dyglib_root / "processed_data" / dataset_name / f"ml_{dataset_name}.csv"
    df = pd.read_csv(path)

    val_time, test_time = np.quantile(
        df["ts"].to_numpy(),
        [(1 - val_ratio - test_ratio), (1 - test_ratio)],
    )

    masks = {
        "train": df["ts"] <= val_time,
        "val": (df["ts"] > val_time) & (df["ts"] <= test_time),
        "test": df["ts"] > test_time,
    }

    print("\n==== DyGLib chronological split statistics ====")
    print(f"val_time={val_time}, test_time={test_time}")
    for split_name, mask in masks.items():
        part = df.loc[mask]
        pos = int(part["label"].sum())
        total = len(part)
        print(
            f"{split_name:>5}: "
            f"#events={total:>8}, #pos={pos:>6}, "
            f"pos_ratio={pos / max(total, 1):.8f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to JODIE data/mooc.csv")
    parser.add_argument("--dyglib_root", type=str, required=True, help="Path to DyGLib root")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mooc",
        help="Use 'mooc' for minimum code changes, or 'mooc_jodie' to avoid overwriting official processed MOOC.",
    )
    parser.add_argument("--node_feat_dim", type=int, default=172)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    args = parser.parse_args()

    csv_path = Path(args.input).expanduser().resolve()
    dyglib_root = Path(args.dyglib_root).expanduser().resolve()

    df = load_jodie_mooc(csv_path)
    check_mooc_semantics(df)
    build_dyglib_processed(
        df=df,
        dyglib_root=dyglib_root,
        dataset_name=args.dataset_name,
        node_feat_dim=args.node_feat_dim,
    )
    print_dyglib_split_statistics(
        dyglib_root=dyglib_root,
        dataset_name=args.dataset_name,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()
