"""
Statistics used in the paper.

Frames of one BSTRB video share a single transcript, so they are not independent samples.
Every interval and test therefore resamples whole units -- a video for video frames, the image
itself for photos and for the general OCR sets -- and recomputes the item-weighted mean
(sum of the resampled units' item values / number of resampled items).

  cluster_bootstrap_mean  mean with a 95% unit-bootstrap CI
  paired_test             mean paired difference (a - b), 95% CI, two-sided bootstrap p
  holm                    Holm-Bonferroni step-down adjustment across a family of tests

p-values are 2 x min(share of bootstrap means <= 0, share >= 0), floored at 1/B; a p at the
floor is reported as "p < 1/B".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Hashable, List, Optional, Sequence

import numpy as np

DEFAULT_B = 4000


def _unit_index(units: Sequence[Hashable]):
    uniq = {u: k for k, u in enumerate(dict.fromkeys(units))}
    return np.array([uniq[u] for u in units]), len(uniq)


def _boot_means(values: np.ndarray, units: Sequence[Hashable], B: int, rng) -> np.ndarray:
    iu, k = _unit_index(units)
    usum = np.bincount(iu, values, k)
    ucnt = np.bincount(iu, None, k)
    draw = rng.integers(0, k, size=(B, k))
    return usum[draw].sum(1) / ucnt[draw].sum(1)


@dataclass
class MeanCI:
    n: int
    units: int
    mean: float
    lo: float
    hi: float


def cluster_bootstrap_mean(values: Sequence[float], units: Sequence[Hashable], *, B: int = DEFAULT_B,
                           seed: int = 0) -> MeanCI:
    v = np.asarray(values, float)
    boot = _boot_means(v, units, B, np.random.default_rng(seed))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return MeanCI(len(v), len(set(units)), float(v.mean()), float(lo), float(hi))


@dataclass
class PairedResult:
    n: int
    units: int
    mean_a: float
    mean_b: float
    diff: float
    lo: float
    hi: float
    p: float
    p_at_floor: bool
    p_holm: Optional[float] = None
    verdict: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def paired_test(a: Sequence[float], b: Sequence[float], units: Sequence[Hashable], *,
                B: int = DEFAULT_B, seed: int = 0, rng=None) -> PairedResult:
    """Paired comparison of two models scored on the same items (a - b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = rng if rng is not None else np.random.default_rng(seed)
    boot = _boot_means(d, units, B, rng)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    raw = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return PairedResult(n=len(d), units=len(set(units)), mean_a=float(a.mean()), mean_b=float(b.mean()),
                        diff=float(d.mean()), lo=float(lo), hi=float(hi), p=max(float(raw), 1 / B),
                        p_at_floor=bool(raw < 1 / B))


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def apply_holm(results: List[PairedResult], alpha: float = 0.05) -> List[PairedResult]:
    """Fill p_holm and a verdict ('better' / 'worse' / 'n.s.') for a family of tests."""
    for r, ph in zip(results, holm([r.p for r in results])):
        r.p_holm = ph
        r.verdict = ("better" if r.lo > 0 and ph < alpha else
                     "worse" if r.hi < 0 and ph < alpha else "n.s.")
    return results


__all__ = ["cluster_bootstrap_mean", "paired_test", "holm", "apply_holm", "PairedResult", "MeanCI"]
