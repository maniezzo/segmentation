from pathlib import Path
import pandas as pd

def sampleM3(
    base_dir=".",
    m3_subdir="M3",
    file_name="M3month.csv",
    n_random=4,
    random_state=42,
    output_csv=None,
):
    base_dir = Path(base_dir)
    m3_dir   = base_dir / m3_subdir
    path     = m3_dir / file_name

    df = pd.read_csv(path)

    # time-series value columns
    value_cols = [c for c in df.columns if str(c).strip().isdigit()]
    # Sort (1..144), just in case
    value_cols = sorted(value_cols, key=lambda x: int(str(x).strip()))

    # Count non-null values (series length)
    df["non_null_count"] = df[value_cols].notna().sum(axis=1)

    rng = random_state

    def pick_group(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) == 0:
            return g

        # Longest series
        max_len = g["non_null_count"].max()
        longest_candidates = g[g["non_null_count"] == max_len]
        longest = longest_candidates.sample(n=1, random_state=rng)

        # Remaining pool for random picks
        remaining = g.drop(index=longest.index)

        k = min(n_random, len(remaining))
        random_pick = remaining.sample(n=k, random_state=rng) if k > 0 else remaining.head(0)

        # Return 4 random + 1 longest (order doesn't matter; keep longest last like before)
        return pd.concat([random_pick, longest], axis=0)

    sampled = (
        df.groupby("Category", group_keys=False)
          .apply(pick_group)
          .reset_index(drop=True)
    )

    # column order (metadata, non_null_count, then values)
    meta_cols = [c for c in df.columns if c not in value_cols and c != "non_null_count"]
    sampled   = sampled[meta_cols + ["non_null_count"] + value_cols]

    if output_csv:
        out_path = Path(output_csv)
        if not out_path.is_absolute():
            out_path = base_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sampled.to_csv(out_path, index=False)

    return sampled

if __name__ == "__main__":
    out = sampleM3(
        base_dir=".",
        m3_subdir="M3",
        file_name="M3month.csv",
        n_random=4,
        random_state=42,
        output_csv="M3sample.csv",
    )
    print("Rows per category:\n", out["Category"].value_counts())
    print("Total rows:", len(out))
    print(out.head())