"""Draw one index from a categorical given in log space, without the failure modes of
``torch.multinomial``.

Why this exists. On the CPU, ``torch.multinomial(p, 1)`` without replacement runs an exponential
race: every entry is divided by an ``Exp(1)`` draw and the largest quotient wins. An exponential
draw that comes back exactly zero turns its entry into infinity, and that entry wins whatever its
probability was. Over a 128k-token vocabulary that is one uniformly random token roughly every
hundred and thirty steps, which is invisible in a unit test on ten categories and ruinous in a
forty-token answer: a fifth of the words in a sampled sequence can be tokens the model gave no
real mass to, and the sequence's own log-probability drops by hundreds of nats.

Inverse-CDF sampling in float64 has no such path. A zero-probability entry has the same cumulative
value as its predecessor, so it can never be the first index whose cumulative value exceeds the
uniform draw. The uniform comes from torch's global CPU generator, so seeding is unchanged.
"""

from __future__ import annotations

import torch


def draw_log(log_weights: torch.Tensor) -> int:
    """One index with probability proportional to ``exp(log_weights)``; ``-1`` if none is finite.

    Entries at ``-inf`` have probability zero and are never returned. Normalization is implicit,
    so the weights need not sum to one.
    """
    prepared = prepare_log(log_weights)
    return -1 if prepared is None else draw_prepared(prepared)


def prepare_log(log_weights: torch.Tensor):
    """The cumulative table :func:`draw_prepared` samples from, or ``None`` if nothing is finite.

    Split out so a caller drawing repeatedly from weights that never change can pay for the
    full-vocabulary pass once. The draw is the same one :func:`draw_log` makes, uniform for uniform.
    """
    lw = log_weights.detach().reshape(-1).to("cpu", torch.float64)
    finite = torch.isfinite(lw)
    if not bool(finite.any()):
        return None
    shift = lw[finite].max()
    probs = torch.where(finite, torch.exp(lw - shift), torch.zeros_like(lw))
    return torch.cumsum(probs, 0), probs


def draw_prepared(prepared) -> int:
    cdf, probs = prepared
    total = cdf[-1]
    u = torch.rand((), dtype=torch.float64) * total  # in [0, total): rand() < 1
    idx = int(torch.searchsorted(cdf, u.reshape(1), right=True))
    if idx >= cdf.shape[0] or probs[idx] <= 0.0:
        idx = int(torch.nonzero(probs > 0.0)[-1])
    return idx


def draw_probs(probs: torch.Tensor) -> int:
    """Same, from probabilities rather than log-weights. Zeros are never returned."""
    p = probs.detach().reshape(-1).to("cpu", torch.float64)
    return draw_log(torch.where(p > 0.0, torch.log(p), torch.full_like(p, float("-inf"))))
