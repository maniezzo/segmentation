from pathlib import Path
import pandas as pd

def sampleM4(
    base_dir=".",
    m4_subdir="M4",
    info_file="M4-info.csv",
    train_file="M4-Quarterly-train.csv",
    min_non_null=100,
    random_state=42,
    output_csv=None,
):
    base_dir = Path(base_dir)
    m4_dir   = base_dir / m4_subdir

    info  = pd.read_csv(m4_dir / info_file)
    train = pd.read_csv(m4_dir / train_file)

    # join keys
    info["M4id"] = info["M4id"].astype(str)
    train["V1"]  = train["V1"].astype(str)

    # Restrict to quarterly
    info_q = info[info["Frequency"].astype(str).str.lower().eq("quarterly")].copy()
    if len(info_q) == 0:
        info_q = info.copy()

    train_q = train[train["V1"].isin(set(info_q["M4id"]))].copy()

    # Merge the two csv
    merged = info_q.merge(train_q, left_on="M4id", right_on="V1", how="inner")

    # the two id columns
    v_cols      = [c for c in train.columns if c.startswith("V")]
    series_cols = [c for c in v_cols if c != "V1"]

    # Count non-null values (series length)
    merged["non_null_count"] = merged[series_cols].notna().sum(axis=1)

    # Filter by minimum length
    merged = merged[merged["non_null_count"] >= min_non_null].copy()

    rng = random_state  # used for reproducibility

    def pick_group(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) == 0:
            return g

        # Longest (break ties randomly but reproducibly)
        max_len = g["non_null_count"].max()
        longest_candidates = g[g["non_null_count"] == max_len]
        longest = longest_candidates.sample(n=1, random_state=rng)

        # Remaining pool for random picks
        remaining = g.drop(index=longest.index)

        # Pick 4 random (or fewer if not enough)
        k = min(4, len(remaining))
        random_pick = remaining.sample(n=k, random_state=rng) if k > 0 else remaining.head(0)

        return pd.concat([random_pick, longest], axis=0)

    sampled = (
        merged.groupby("category", group_keys=False)
        .apply(pick_group)
        .reset_index(drop=True)
    )

    # Reorder columns nicely
    base_info_cols = ["M4id", "category", "Frequency", "Horizon", "SP", "StartingDate"]
    ordered_cols = base_info_cols + ["non_null_count"] + v_cols
    extras = [c for c in sampled.columns if c not in ordered_cols]
    sampled = sampled[ordered_cols + extras]

    if output_csv:
        out_path = Path(output_csv)
        if not out_path.is_absolute():
            out_path = base_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sampled.to_csv(out_path, index=False)

    return sampled


if __name__ == "__main__":
    df = sampleM4(
        base_dir=".",
        min_non_null=100,
        random_state=42,
        output_csv="M4sample.csv",
    )

    print("Rows per category:\n", df["category"].value_counts())
    print("Total rows:", len(df))