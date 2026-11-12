"""The categorical draw must never return a token the distribution gave no real mass to.

This is the test that would have caught the junk-token bug: one uniformly random token every ~130
draws over a 128k vocabulary. Ten categories never show it, so the vocabulary here is real-sized.

Run with ``pytest tests/test_draw.py``; no model and no GPU are involved.
"""

import math

import torch

from casa.utils.draw import draw_log, draw_probs

V = 128256
PLAUSIBLE = (5, 17, 99, 1234)


def _peaked():
    """A next-token distribution shaped like a language model's mid-answer: four plausible tokens
    and a 128k-entry tail at about e^-40 each, which in float64 sums to under 1e-9."""
    torch.manual_seed(0)
    lp = torch.full((V,), -40.0) + torch.randn(V) * 3.0
    lp[5], lp[17], lp[99], lp[1234] = 0.0, -4.5, -4.5, -6.0
    return torch.log_softmax(lp, dim=-1)


def test_never_draws_a_negligible_token():
    lp = _peaked()
    # The precondition is computed in float64: float32 softmax alone leaves a rounding tail near
    # 1e-6, which is not the mass the draw is being tested against.
    tail = float(1.0 - torch.softmax(lp.double(), -1)[list(PLAUSIBLE)].sum())
    assert tail < 1e-9, tail  # the test is only meaningful if the tail really is negligible
    torch.manual_seed(1)
    bad = sum(1 for _ in range(5000) if draw_log(lp) not in PLAUSIBLE)
    assert bad == 0, f"{bad} draws of tokens carrying {tail:.1e} total mass"


def test_frequencies_match_the_distribution():
    lp = torch.full((V,), float("-inf"))
    lp[10], lp[20], lp[30] = math.log(0.7), math.log(0.2), math.log(0.1)
    torch.manual_seed(2)
    n = 30000
    counts = {10: 0, 20: 0, 30: 0}
    for _ in range(n):
        counts[draw_log(lp)] += 1
    for idx, p in ((10, 0.7), (20, 0.2), (30, 0.1)):
        se = math.sqrt(p * (1 - p) / n)
        assert abs(counts[idx] / n - p) < 5 * se, (idx, counts[idx] / n, p)


def test_zero_probability_entries_are_skipped():
    p = torch.tensor([0.0, 0.5, 0.0, 0.5, 0.0])
    torch.manual_seed(3)
    seen = {draw_probs(p) for _ in range(2000)}
    assert seen == {1, 3}, seen


def test_all_impossible_returns_minus_one():
    assert draw_log(torch.full((100,), float("-inf"))) == -1


def test_single_live_entry_at_the_end():
    lp = torch.full((V,), float("-inf"))
    lp[-1] = 0.0
    assert draw_log(lp) == V - 1


def test_unnormalized_weights_are_fine():
    lp = torch.full((V,), float("-inf"))
    lp[7] = math.log(3.0)
    lp[8] = math.log(1.0)
    torch.manual_seed(4)
    n = 20000
    sevens = sum(1 for _ in range(n) if draw_log(lp) == 7)
    assert abs(sevens / n - 0.75) < 0.02


def test_seed_reproduces_the_stream():
    lp = _peaked()
    torch.manual_seed(5)
    a = [draw_log(lp) for _ in range(50)]
    torch.manual_seed(5)
    b = [draw_log(lp) for _ in range(50)]
    assert a == b
