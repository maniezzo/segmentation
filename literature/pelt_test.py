import numpy as np
from typing import Callable


def pelt(data: np.ndarray, cost_fn: Callable, penalty: float) -> list[int]:
    """
    PELT changepoint detection.

    Parameters
    ----------
    data     : 1-D array of observations
    cost_fn  : cost_fn(s, t) returns the cost C(s+1..t) of the segment (s, t]
               Must satisfy superadditivity for the dominance rule to be valid.
    penalty  : per-changepoint penalty (beta)

    Returns
    -------
    List of changepoint positions (0-based, exclusive right endpoint).
    E.g. [100, 200] means changepoints after indices 99 and 199.
    """
    n = len(data)

    # F[t] = optimal penalised cost for data[0..t-1]
    # F[0] = -penalty so that the s=0 candidate (no changepoint yet) contributes
    #   F[0] + C(0, t) + penalty = C(0, t)   — a single segment with no penalty.
    F = np.full(n + 1, np.inf)
    F[0] = -penalty

    # last[t] = the last changepoint position in the optimal partition ending at t
    last = np.zeros(n + 1, dtype=int)

    # Active set: candidate last-changepoint positions not yet pruned.
    # s in active means "a segment starting at s+1 is still worth considering".
    active = [0]

    for t in range(1, n + 1):

        # ------------------------------------------------------------------
        # Single fused pass: compute F[s] + C(s, t) for every active s,
        # find the best, and collect survivors — all without re-evaluating
        # cost_fn a second time.
        # ------------------------------------------------------------------
        costs = [(F[s] + cost_fn(s, t), s) for s in active]

        # Best raw cost (without the per-changepoint penalty)
        best_raw, best_s = min(costs)

        # Store the optimal penalised cost for use in future steps
        F[t]    = best_raw + penalty
        last[t] = best_s

        # ------------------------------------------------------------------
        # Dominance pruning (superadditivity required):
        #   prune s if  F[s] + C(s, t)  >  best_raw
        # i.e. keep only candidates whose raw cost does not exceed the best.
        # Threshold and costs are on the same scale (no penalty added to either).
        # ------------------------------------------------------------------
        active = [s for raw, s in costs if raw <= best_raw]

        # Add t as a new candidate for future segments
        active.append(t)

    # ------------------------------------------------------------------
    # Backtrack through `last` to recover changepoint positions
    # ------------------------------------------------------------------
    changepoints = []
    t = n
    while t > 0:
        cp = last[t]
        if cp > 0:
            changepoints.append(cp)
        t = cp

    changepoints.reverse()
    return changepoints


# ── Squared-error segment cost (closed-form, O(1) via prefix sums) ────────────

def make_seg_cost(data: np.ndarray) -> Callable:
    """
    Returns a closure cost(s, t) = TSS of data[s:t]  (total sum of squares
    around the segment mean).  Satisfies superadditivity, so the PELT
    dominance rule is valid.

    C(s, t) = sum_{i=s}^{t-1} (x_i - mean(x_s..x_{t-1}))^2
            = sum_sq[s:t] - (sum[s:t])^2 / (t - s)
    """
    prefix    = np.concatenate([[0.0], np.cumsum(data)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(data ** 2)])

    def cost(s: int, t: int) -> float:
        n_seg = t - s
        if n_seg <= 0:
            return 0.0
        su = prefix[t]    - prefix[s]
        sq = prefix_sq[t] - prefix_sq[s]
        return sq - su ** 2 / n_seg

    return cost


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Three segments with means 0, 5, 2 — true changepoints at 100 and 200
    data = np.concatenate([
        rng.normal(0, 1, 100),
        rng.normal(5, 1, 100),
        rng.normal(2, 1, 100),
    ])

    cost_fn      = make_seg_cost(data)
    changepoints = pelt(data, cost_fn, penalty=10.0)

    print("Detected changepoints:", changepoints)
    # Expected output: [100, 200]