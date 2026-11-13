"""MARS draws exactly from its target. Checked without a language model or a GPU.

The main instrument is the arithmetic-expression example from the MARS paper: two tiny models over
the alphabet {0, 1, +} whose next-token distribution depends only on the last token. The support is
finite, so the exact target of every operation can be enumerated and compared with what the
sampler produces, in total variation against the noise floor of a sample of the same size.

Also here: which expressions the algebra accepts, the runtime guards for expressions that are not
really envelopes, the recognizer bridge grammars go through, float32 tolerances at a real
vocabulary size, and regressions found by review.

Run with ``pytest tests/test_mars_exact.py``.
"""

import ast
import math
from collections import Counter
from pathlib import Path

import pytest
import torch

import casa.algebra
from casa.algebra import (Model, NoEnvelope, Scorer, agree, constrain, intersect, mean, reweight,
                          union)
from casa.samplers.mars import MARS, _LOG_SLACK
from casa.utils import envelope_runtime as rt
from casa.utils.envelope_runtime import _RecognizerMask

# =============================================================================================
# The tabular models
# =============================================================================================

ZERO, ONE, PLUS, EOS, BOS = 0, 1, 2, 3, 4
VOCAB = 5
MAX_BODY = 5  # after this many tokens both models emit end-of-sequence with probability one

P_TABLE = {
    "start": [0.4, 0.3, 0.3, 0.0, 0.0],
    "digit": [0.1, 0.1, 0.5, 0.3, 0.0],
    "plus": [0.5, 0.4, 0.1, 0.0, 0.0],
}
R_TABLE = {
    "start": [0.5, 0.5, 0.0, 0.0, 0.0],
    "digit": [0.0, 0.0, 0.4, 0.6, 0.0],
    "plus": [0.5, 0.5, 0.0, 0.0, 0.0],
}
STOP = [0.0, 0.0, 0.0, 1.0, 0.0]
# The language 0(+0)*, as a table, so a constrained target can be enumerated.
MASK_TABLE = {
    "start": [1.0, 0.0, 0.0, 0.0, 0.0],
    "digit": [0.0, 0.0, 1.0, 1.0, 0.0],
    "plus": [1.0, 0.0, 0.0, 0.0, 0.0],
}


def state_of(ctx):
    if not ctx:
        return "start"
    return "plus" if ctx[-1] == PLUS else "digit"


def row(table, ctx):
    return STOP if len(ctx) >= MAX_BODY else table[state_of(ctx)]


class FakeTokenizer:
    eos_token_id = EOS
    _vocab = {"0": ZERO, "1": ONE, "+": PLUS, "$": EOS, "<s>": BOS}

    def __len__(self):
        return VOCAB

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=False):
        # A one-token prompt. A forward pass gives the conditional for a token only from the
        # position before it, so the first generated token needs something to its left.
        return [BOS]

    def decode(self, ids):
        inv = {v: k for k, v in self._vocab.items()}
        return "".join(inv[i] for i in ids if i != EOS)


class FakeModel:
    """Table logits at every position, as a real causal model returns."""

    device = torch.device("cpu")

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        self.calls += 1
        ids = input_ids[0].tolist()
        # Position j predicts token j+1 given ids[:j+1]; ids[0] is the prompt token.
        rows = [row(self.table, ids[1:j + 1]) for j in range(len(ids))]
        out = type("Out", (), {})()
        logits = torch.log(torch.tensor(rows, dtype=torch.float32).clamp_min(1e-30))
        out.logits = logits.unsqueeze(0)
        out.past_key_values = None
        return out


class FakeLLM:
    def __init__(self, table, tokenizer):
        self.model = FakeModel(table)
        self.tokenizer = tokenizer
        self.model_id = "fake"

    def format_prompt(self, prompt):
        return ""


def models():
    """Fresh P and R over a shared tokenizer."""
    tok = FakeTokenizer()
    return Model(FakeLLM(P_TABLE, tok), name="P"), Model(FakeLLM(R_TABLE, tok), name="R")


def enumerate_target(op, table_a=P_TABLE, table_b=R_TABLE, post=None):
    """Exact weights of every complete sequence, as a dict from token tuple to weight."""
    out = {}

    def walk(ctx, lp, lr):
        pr, rr = row(table_a, ctx), row(table_b, ctx)
        for tok in range(VOCAB):
            p, r = pr[tok], rr[tok]
            if p <= 0 and r <= 0:
                continue
            np_, nr = lp * p, lr * r
            if tok == EOS:
                w = op(np_, nr)
                if post is not None:
                    w *= post(tuple(ctx))
                if w > 0:
                    out[tuple(ctx)] = out.get(tuple(ctx), 0.0) + w
            elif len(ctx) < MAX_BODY and (np_ > 0 or nr > 0):
                walk(ctx + [tok], np_, nr)

    walk([], 1.0, 1.0)
    return out


def total_variation(emp, exact):
    keys = set(emp) | set(exact)
    return 0.5 * sum(abs(emp.get(k, 0.0) - exact.get(k, 0.0)) for k in keys)


def noise_floor(exact, n):
    """Expected total variation between the exact law and an n-sample draw *from it*.

    For each outcome the absolute deviation has mean about sqrt(2 p (1-p) / (pi n)), so the
    expected total variation is half their sum. A fixed tolerance would instead pass a broken
    sampler on a small support or fail a correct one on a large support.
    """
    return 0.5 * sum(math.sqrt(2 * p * (1 - p) / (math.pi * n)) for p in exact.values())


N = 12_000
SLACK = 3.0


def draw(target, op, n=N, seed=0, tables=(P_TABLE, R_TABLE), post=None, adaptive=True,
         cache=False):
    """Sample ``n`` times and return (TV to the exact law, tolerance, sampler, samples)."""
    torch.manual_seed(seed)
    sampler = MARS(target, max_new_tokens=MAX_BODY + 1, adaptive=adaptive, cache=cache)
    results = sampler.sample("", n_samples=n, max_attempts=10_000)
    counts = Counter(tuple(r.token_ids[:-1]) for r in results)
    emp = {k: v / len(results) for k, v in counts.items()}
    weights = enumerate_target(op, *tables, post=post)
    z = sum(weights.values())
    exact = {k: v / z for k, v in weights.items()}
    return total_variation(emp, exact), SLACK * noise_floor(exact, n), sampler, results


def assert_exact(target, op, **kw):
    tv, tol, sampler, results = draw(target, op, **kw)
    assert len(results) == kw.get("n", N), f"only {len(results)} samples"
    assert tv < tol, f"TV {tv:.4f} exceeds tolerance {tol:.4f}"
    return sampler


def geomean(a, b):
    return math.sqrt(a * b)


# =============================================================================================
# The algebra: which expressions admit an envelope
# =============================================================================================

class _Tag:
    """Stand-in for an LLM. The algebra never looks inside one."""

    def __init__(self, tag):
        self.tag = tag


PA, RA, SA = Model(_Tag("a"), name="P"), Model(_Tag("b"), name="R"), Model(_Tag("c"), name="S")


def _mask(ctx, vocab_size):
    return [0.0] * vocab_size  # everything allowed; the algebra never calls this


def test_algebra_imports_only_the_standard_library():
    # The algebra is about expressions, not tensors; keeping it free of torch keeps it checkable
    # anywhere. This fails if it ever grows a heavier import.
    tree = ast.parse(Path(casa.algebra.__file__).read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert not roots & {"torch", "numpy", "transformers", "casa"}, roots


@pytest.mark.parametrize("label, make", [
    ("base", lambda: PA),
    ("product", lambda: PA * RA),
    ("intersect as powers", lambda: PA ** 0.5 * RA ** 0.5),
    ("intersect", lambda: intersect(PA, RA)),
    ("sharpen", lambda: PA ** 2),
    ("agree, min", lambda: PA & RA),
    ("union, mixture", lambda: PA | RA),
    ("harmonic mean", lambda: mean([PA, RA], tau=-1)),
    ("mean tau=0", lambda: mean([PA, RA], tau=0)),
    ("mean tau=1", lambda: mean([PA, RA], tau=1)),
    ("constrain (CARS)", lambda: constrain(PA, _mask)),
    ("constrain an ensemble", lambda: constrain(PA * RA, _mask)),
    ("prefix-monotone scorer",
     lambda: PA * Scorer(lambda c: 1.0, prefix_monotone=True, log_mask=_mask)),
    ("exponents collected: .5 + .7", lambda: PA ** 0.5 * PA ** 0.7),
    ("(P^.5 R^.5)^2", lambda: (PA ** 0.5 * RA ** 0.5) ** 2),
    ("sharpen a constrained model", lambda: constrain(PA, _mask) ** 2),
    ("constrained model times another", lambda: constrain(PA, _mask) * RA),
    ("plain min needs no leaf step", lambda: PA & SA),
])
def test_expressions_that_admit_an_envelope(label, make):
    env = make().envelope()
    assert env.explain().strip()


@pytest.mark.parametrize("label, make, leaf", [
    ("quadratic mean", lambda: mean([PA, RA], tau=2), 0.5),
    ("maximum", lambda: mean([PA, RA], tau=math.inf), 0.5),
    ("quadratic, weights .25/.75", lambda: mean([PA, RA], tau=2, weights=[0.25, 0.75]), 0.25),
    ("output-only verifier", lambda: reweight(PA, lambda ctx: 0.5, prefix_monotone=False), 0.0),
    ("min(max(P,R), S)",
     lambda: mean([mean([PA, RA], tau=math.inf), SA], tau=-math.inf), 0.5),
    ("max(P,R) * S", lambda: mean([PA, RA], tau=math.inf) * SA, 0.5),
    ("mix(quadratic(P,R), S)", lambda: mean([mean([PA, RA], tau=2), SA], tau=1), 0.5),
])
def test_expressions_that_need_a_leaf_step(label, make, leaf):
    # A leaf requirement must reach the root however deeply it is nested.
    env = make().envelope()
    assert env.leaf_acceptance is not None, env.explain()
    assert math.isclose(env.leaf_acceptance, leaf), env.explain()


@pytest.mark.parametrize("label, make", [
    ("temper", lambda: PA ** 0.5),
    ("temper inside a product", lambda: PA ** 0.3 * RA ** 0.3),
    ("temper twice", lambda: (PA ** 0.5) ** 0.5),
    ("exponents collected: .3 + .3", lambda: PA ** 0.3 * PA ** 0.3),
    ("temper a constrained model", lambda: constrain(PA, _mask) ** 0.5),
])
def test_expressions_with_no_envelope_are_refused(label, make):
    with pytest.raises(NoEnvelope):
        make().envelope()


def test_contrast_is_refused_at_construction():
    with pytest.raises(NoEnvelope):
        PA / RA


def test_within_model_ensembles_are_expressions_too():
    llm = _Tag("shared")
    p1 = Model(llm, name="P", prompt="My favorite physicist is")
    p2 = Model(llm, name="P", prompt="My favorite author is")
    (p1 & p2).envelope()
    (p1 ** 0.5 * p2 ** 0.5).envelope()


def test_explain_is_text():
    for expr in (intersect(PA, RA), agree(PA, RA), union(PA, RA), mean([PA, RA], tau=math.inf)):
        assert expr.envelope().explain().strip()


# =============================================================================================
# Exactness against the enumerated target
# =============================================================================================

@pytest.mark.parametrize("label, make, op", [
    ("intersect, sqrt(P R)", lambda p, r: intersect(p, r), geomean),
    ("product, P R", lambda p, r: p * r, lambda a, b: a * b),
    ("agree, min(P, R)", lambda p, r: p & r, min),
    ("union, mixture", lambda p, r: p | r, lambda a, b: 0.5 * a + 0.5 * b),
    ("max(P, R)", lambda p, r: mean([p, r], tau=math.inf), max),
])
def test_ensembles_are_sampled_exactly(label, make, op):
    assert_exact(make(*models()), op)


def test_coverage_regime_folds_the_leaf_gap_into_the_trie():
    # The maximum starts from the dominating mixture, which is tight at every interior prefix, so
    # all of its slack sits at the leaves. MARS must fold that slack into the bound at the
    # expansion, so the root mass falls, rather than flip for it at the leaf forever.
    sampler = assert_exact(mean([*models()], tau=math.inf), max)
    assert sampler.stats.leaf_tightenings > 0
    assert sampler.root_mass < 1.0


def test_single_model_needs_no_rejection():
    p, _ = models()
    sampler = assert_exact(p, lambda a, b: a)
    assert sampler.stats.rejections == 0


def test_hard_constraint_reduces_to_cars():
    # A 0/1 potential under exponent one: the constrained LM distribution, which is CARS's target.
    tok = FakeTokenizer()
    p = Model(FakeLLM(P_TABLE, tok), name="P")
    m = Model(FakeLLM(MASK_TABLE, tok), name="L", sub_stochastic=False)
    assert_exact(p * m, lambda a, b: a * b, tables=(P_TABLE, MASK_TABLE))


def lang_mask(ctx, vocab_size):
    # Only "0", then alternating "+" and "0", ending any time after a digit: 0(+0)*
    allow = {PLUS, EOS} if (ctx and ctx[-1] != PLUS) else {ZERO}
    return torch.tensor([0.0 if t in allow else float("-inf") for t in range(vocab_size)])


def test_prefix_monotone_scorer_is_the_grammar_path():
    p, _ = models()
    assert_exact(constrain(p, lang_mask), lambda a, b: a * b, tables=(P_TABLE, MASK_TABLE))


def test_sharpened_constrained_model():
    p, _ = models()
    assert_exact(constrain(p, lang_mask) ** 2, lambda a, b: a * a * b, tables=(P_TABLE, MASK_TABLE))


def test_output_only_verifier_is_scored_at_the_prefix_it_completes():
    def verifier(ctx):
        return 0.25 if len(ctx) % 2 else 1.0  # depends on the whole sequence

    p, _ = models()
    sampler = assert_exact(reweight(p, verifier, prefix_monotone=False), lambda a, b: a,
                           tables=(P_TABLE, P_TABLE), post=verifier)
    assert sampler.stats.leaf_tightenings > 0, "the verifier was never consulted"


def test_root_mass_never_increases():
    # The root mass is the quantity the theorem says is monotone; the acceptance rate is Z over it.
    torch.manual_seed(2)
    sampler = MARS(intersect(*models()), max_new_tokens=MAX_BODY + 1)
    masses = [sampler.root_mass]
    for i in range(30):
        sampler.sample("", n_samples=1 if i < 10 else 20, max_attempts=10_000)
        masses.append(sampler.root_mass)
    assert all(b <= a + 1e-9 for a, b in zip(masses, masses[1:])), masses
    assert masses[-1] < masses[0] - 1e-9, "never decreased, so the test witnessed nothing"


def test_forced_termination_at_the_bound_is_the_default():
    torch.manual_seed(6)
    stopper = MARS(intersect(*models()), max_new_tokens=2)
    got = stopper.sample("", n_samples=20, max_attempts=500)
    assert len(got) == 20 and stopper.stats.length_cutoffs == 0
    assert max(len(g.token_ids) - 1 for g in got) <= 2


def test_discarding_at_the_bound_does_not_crash():
    torch.manual_seed(5)
    short = MARS(intersect(*models()), max_new_tokens=2, on_max_length="discard")
    short.sample("", n_samples=3, max_attempts=50)
    assert short.stats.length_cutoffs > 0


def test_revisited_prefix_costs_no_model_call():
    torch.manual_seed(4)
    p, r = models()
    sampler = MARS(intersect(p, r), max_new_tokens=MAX_BODY + 1)
    sampler.sample("", n_samples=2000, max_attempts=10_000)
    calls = sampler.stats.model_calls
    assert calls == p.llm.model.calls + r.llm.model.calls
    # Two models, one pass each per expanded node, plus a few leaf-weight queries. If advancing
    # through cached nodes still asked the models this would scale with total descent length.
    assert calls <= 2 * sampler.stats.expansions + 8


def trie_inconsistencies(sampler, tol=1e-6):
    """Edges where the parent's correction for a child differs from the child's own total."""
    bad = []

    def walk(node, path):
        if node.raw_logprob is None:
            return
        for tokid, child in node.children.items():
            if child.raw_logprob is None:
                continue  # created but never expanded; the parent still holds the original bound
            parent_says = float(torch.exp(node.log_theta[0, tokid] if node.log_theta is not None
                                          else torch.tensor(0.0)))
            child_total = float(torch.exp(torch.logsumexp(
                child.raw_logprob[0] + (child.log_theta[0] if child.log_theta is not None else 0.0),
                dim=0)))
            if abs(parent_says - child_total) > tol * max(1.0, abs(child_total)):
                bad.append((path + [tokid], parent_says, child_total))
            walk(child, path + [tokid])

    walk(sampler.trie.root, [])
    return bad


@pytest.mark.parametrize("label, make, seed", [
    ("intersect", lambda p, r: intersect(p, r), 10),
    ("max, leaf rejection", lambda p, r: mean([p, r], tau=math.inf), 11),
])
def test_every_parent_bound_equals_its_childs_total(label, make, seed):
    torch.manual_seed(seed)
    sampler = MARS(make(*models()), max_new_tokens=MAX_BODY + 1)
    sampler.sample("", n_samples=1500, max_attempts=10_000)
    assert trie_inconsistencies(sampler) == []


# =============================================================================================
# The non-adaptive modes: plain and cached rejection sampling from the same envelope
# =============================================================================================

def test_plain_rejection_sampling_hits_the_same_distribution_and_does_not_amortize():
    _, _, rs, _ = draw(intersect(*models()), geomean, adaptive=False)
    tv, tol, _, results = draw(intersect(*models()), geomean, adaptive=False)
    assert len(results) == N and tv < tol
    _, _, mars, _ = draw(intersect(*models()), geomean)
    # RS clears the trie before each descent and pays afresh; MARS pays once per prefix.
    rs_per_descent = rs.stats.model_calls / max(rs.stats.descents, 1)
    mars_per_descent = mars.stats.model_calls / max(mars.stats.descents, 1)
    assert mars_per_descent < rs_per_descent / 10


def test_cached_rejection_sampling_pays_per_prefix_but_learns_nothing():
    for make, op in [(lambda p, r: intersect(p, r), geomean),
                     (lambda p, r: mean([p, r], tau=math.inf), max)]:
        tv, tol, _, results = draw(make(*models()), op, adaptive=False, cache=True)
        assert len(results) == N and tv < tol
    _, _, rs, _ = draw(intersect(*models()), geomean, adaptive=False)
    _, _, rsc, _ = draw(intersect(*models()), geomean, adaptive=False, cache=True)
    _, _, mars, _ = draw(intersect(*models()), geomean)
    rs_pd = rs.stats.model_calls / max(rs.stats.descents, 1)
    rsc_pd = rsc.stats.model_calls / max(rsc.stats.descents, 1)
    assert rsc_pd < rs_pd / 10
    assert abs(rsc.acceptance_rate - rs.acceptance_rate) < 0.03
    assert rsc.acceptance_rate < mars.acceptance_rate - 0.05
    assert rsc.trie.root.log_theta is None  # no bound was ever tightened


def test_cached_rejection_sampling_draws_the_same_samples_as_plain():
    # The cache changes only where the ratios come from, never which coins are flipped.
    draws = []
    for cache in (False, True):
        torch.manual_seed(7)
        s = MARS(mean([*models()], tau=math.inf), max_new_tokens=MAX_BODY + 1,
                 adaptive=False, cache=cache)
        got = s.sample("", n_samples=500, max_attempts=10_000)
        draws.append(([tuple(x.token_ids) for x in got], s.stats.descents, s.stats.model_calls))
    assert draws[0][:2] == draws[1][:2]
    assert draws[1][2] < draws[0][2]


def test_cache_requires_non_adaptive():
    with pytest.raises(ValueError):
        MARS(intersect(*models()), cache=True)


# =============================================================================================
# Runtime guards: expressions the algebra accepts but the numbers contradict
# =============================================================================================

GV, GEOS, GBOS = 4, 2, 3


class GuardTok:
    eos_token_id = GEOS

    def __len__(self):
        return GV

    def get_vocab(self):
        return {"a": 0, "b": 1, "$": GEOS, "<s>": GBOS}

    def encode(self, text, add_special_tokens=False):
        return [GBOS]

    def decode(self, ids):
        return "".join("ab$<s>"[i] for i in ids)


class FlatLLM:
    """Every position returns the same row of weights."""

    def __init__(self, weights, tokenizer):
        self.tokenizer = tokenizer
        self.model_id = "fake"
        w = weights

        def model(input_ids, past_key_values=None, use_cache=False):
            rows = torch.tensor([w] * input_ids.shape[-1], dtype=torch.float32)
            out = type("Out", (), {})()
            out.logits = torch.log(rows.clamp_min(1e-30)).unsqueeze(0)
            out.past_key_values = None
            return out

        model.device = torch.device("cpu")
        self.model = model

    def format_prompt(self, prompt):
        return ""


def test_condition_ii_violation_is_caught_at_the_first_expansion():
    # Declared not sub-stochastic, so passed through unnormalized, and every entry is 2.0. The
    # algebra accepts it under an exponent of one; the runtime must not.
    tok = GuardTok()
    good = Model(FlatLLM([0.3, 0.3, 0.4, 0.0], tok), name="P")
    bad = Model(FlatLLM([2.0, 2.0, 2.0, 0.0], tok), name="B", sub_stochastic=False)
    with pytest.raises(ValueError, match="condition \\(ii\\) violated"):
        MARS(good * bad, max_new_tokens=4).sample("", n_samples=1, max_attempts=5)


def test_bounded_potential_is_fine():
    tok = GuardTok()
    good = Model(FlatLLM([0.3, 0.3, 0.4, 0.0], tok), name="P")
    ok = Model(FlatLLM([0.5, 0.5, 0.5, 0.0], tok), name="B", sub_stochastic=False)
    assert len(MARS(good * ok, max_new_tokens=4).sample("", n_samples=5, max_attempts=200)) == 5


def test_empty_prompt_is_refused():
    class NoPromptTok(GuardTok):
        def encode(self, text, add_special_tokens=False):
            return []

    bare = Model(FlatLLM([0.3, 0.3, 0.4, 0.0], NoPromptTok()), name="P")
    with pytest.raises(ValueError, match="empty prompt"):
        MARS(bare, max_new_tokens=4).sample("", n_samples=1)


def test_mixing_token_spaces_is_refused():
    other = GuardTok()
    other.get_vocab = lambda: {"x": 0, "y": 1, "$": GEOS, "<s>": GBOS}
    a = Model(FlatLLM([0.3, 0.3, 0.4, 0.0], GuardTok()), name="P")
    b = Model(FlatLLM([0.3, 0.3, 0.4, 0.0], other), name="R")
    with pytest.raises(Exception, match="token space"):
        MARS(a * b, max_new_tokens=4).sample("", n_samples=1)


# =============================================================================================
# A real vocabulary size: float32 arithmetic must fit inside the tolerances
# =============================================================================================

BIG, EOS_BIG = 152064, 3


class BigTok:
    eos_token_id = EOS_BIG

    def __len__(self):
        return BIG

    def get_vocab(self):
        return {"t%d" % i: i for i in range(8)}  # only compared against itself here

    def encode(self, text, add_special_tokens=False):
        return [7]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class BigLLM:
    def __init__(self, seed, tokenizer, eos_boost=0.0, scale=2.0):
        self.tokenizer = tokenizer
        self.model_id = "big"

        def model(input_ids, past_key_values=None, use_cache=False):
            n = input_ids.shape[-1]
            g = torch.Generator().manual_seed(seed + n)
            logits = torch.randn(n, BIG, generator=g) * scale
            logits[:, EOS_BIG] += eos_boost  # terminate promptly
            out = type("Out", (), {})()
            out.logits = logits.unsqueeze(0)
            out.past_key_values = None
            return out

        model.device = torch.device("cpu")
        self.model = model

    def format_prompt(self, prompt):
        return prompt


def test_production_sized_vocabulary_does_not_trip_the_envelope_guard():
    # Summing a float32 log-softmax over 150k entries is off by about 5e-5; a 1e-6 tolerance on
    # condition (ii) rejected every ordinary model on its first expansion.
    tok = BigTok()
    p = Model(BigLLM(1, tok, eos_boost=14.0), name="P")
    r = Model(BigLLM(2, tok, eos_boost=14.0), name="R")
    got = MARS(p ** 0.5 * r ** 0.5, max_new_tokens=4).sample("hi", n_samples=5, max_attempts=200)
    assert len(got) == 5
    solo = Model(BigLLM(3, tok, eos_boost=14.0), name="S")
    assert len(MARS(solo, max_new_tokens=4).sample("hi", n_samples=5, max_attempts=200)) == 5


def test_float32_error_at_this_size_exceeds_1e_6_but_sits_inside_the_slack():
    # Measured directly, on flat-ish distributions, where many entries contribute to the sum.
    slack = math.expm1(_LOG_SLACK)
    worst = 0.0
    for seed in range(12):
        flat = Model(BigLLM(100 + seed, BigTok(), scale=0.5), name="M")
        ratios = rt.plan(flat.envelope(), {}, 1.0).fresh().log_ratios()
        worst = max(worst, abs(float(torch.exp(ratios.double()).sum()) - 1.0))
    assert worst > 1e-6, worst
    assert worst < slack / 4, (worst, slack)


def test_partial_prompt_override_keeps_the_other_models_prompt():
    class SpyTok(BigTok):
        def __init__(self):
            self.seen = []

        def encode(self, text, add_special_tokens=False):
            self.seen.append(text)
            return [7]

    tok = SpyTok()
    a = Model(BigLLM(4, tok, eos_boost=14.0), name="A")
    b = Model(BigLLM(5, tok, eos_boost=14.0), name="B")
    m = MARS(a ** 0.5 * b ** 0.5, max_new_tokens=3)
    tok.seen.clear()
    m.sample("shared prompt", n_samples=1, max_attempts=50, prompts={id(a): "special"})
    assert "special" in tok.seen and "shared prompt" in tok.seen, tok.seen


# =============================================================================================
# Coverage regime: the reported weight is the target's, and leaves cost no forward pass
# =============================================================================================

SV, SEOS = 4, 2
PT = {"a": [0.3, 0.2, 0.5, 0.0], "b": [0.1, 0.4, 0.5, 0.0]}
RT = {"a": [0.6, 0.1, 0.3, 0.0], "b": [0.2, 0.2, 0.6, 0.0]}
SMAXLEN = 4


def _sstate(ctx):
    return "a" if not ctx or ctx[-1] == 0 else "b"


class SmallTok:
    eos_token_id = SEOS

    def __len__(self):
        return SV

    def get_vocab(self):
        return {"0": 0, "1": 1, "$": SEOS, "<s>": 3}

    def encode(self, text, add_special_tokens=False):
        return [3]

    def decode(self, ids):
        return "".join("01$s"[i] for i in ids)


class SmallLLM:
    def __init__(self, table, tokenizer):
        self.tokenizer = tokenizer
        self.model_id = "small"

        def model(input_ids, past_key_values=None, use_cache=False):
            ids = input_ids[0].tolist()
            rows = [table[_sstate(ids[1:j + 1])] for j in range(len(ids))]
            out = type("Out", (), {})()
            out.logits = torch.log(torch.tensor(rows).clamp_min(1e-30)).unsqueeze(0)
            out.past_key_values = None
            return out

        model.device = torch.device("cpu")
        self.model = model

    def format_prompt(self, prompt):
        return ""


def _weight(table, seq, stopped):
    """Weight of a complete sequence. One stopped at the length bound does not pay for the marker:
    MARS forces termination there, so its weight is the prefix weight."""
    w, ctx = 1.0, []
    for t in list(seq) + ([] if stopped else [SEOS]):
        w *= table[_sstate(ctx)][t]
        ctx.append(t)
    return w


def test_reported_weight_is_the_targets_not_the_dominating_envelopes():
    tok = SmallTok()
    cov = MARS(mean([Model(SmallLLM(PT, tok), name="P"), Model(SmallLLM(RT, tok), name="R")],
                    tau=math.inf), max_new_tokens=SMAXLEN)
    torch.manual_seed(0)
    worst = 0.0
    for s in cov.sample("", n_samples=250, max_attempts=500):
        seq = tuple(s.token_ids[:-1])
        stopped = len(seq) >= SMAXLEN
        want = math.log(max(_weight(PT, seq, stopped), _weight(RT, seq, stopped)))
        worst = max(worst, abs(s.raw_logprob - want))
    assert worst < 1e-4, worst


def test_coverage_regime_pays_no_forward_pass_per_yielded_sequence():
    tok = SmallTok()
    torch.manual_seed(1)
    perf = MARS(mean([Model(SmallLLM(PT, tok), name="P"), Model(SmallLLM(RT, tok), name="R")],
                     tau=math.inf), max_new_tokens=SMAXLEN)
    perf.sample("", n_samples=400, max_attempts=500)
    assert perf.stats.model_calls <= 2 * perf.stats.expansions + 8


# =============================================================================================
# The recognizer bridge grammars go through
# =============================================================================================

RV = 5


class FakeRecognizer:
    """Mimics a CASA token recognizer: sequential, with a consumed-token counter.

    Accepts any sequence that alternates even and odd token ids starting with an even one, which
    is enough structure to tell a correct replay from a broken one.
    """

    vocab_size = RV

    def __init__(self):
        self.current_index = 0
        self.path = []
        self.resets = 0
        self.advanced = 0

    def reset(self):
        self.resets += 1
        self.current_index = 0
        self.path = []

    def try_advance_token_ids(self, token_ids):
        new = token_ids[self.current_index:].tolist()
        self.advanced += len(new)
        cand = self.path + new
        if not all(t % 2 == i % 2 for i, t in enumerate(cand)):
            return False
        self.path = cand
        self.current_index += len(new)
        return True

    def filter_vocab(self):
        want = len(self.path) % 2
        return [t for t in range(RV) if t % 2 == want]

    def apply_token_bitmask(self, logits, bitmask):
        allowed = set(bitmask)
        for t in range(logits.shape[-1]):
            if t not in allowed:
                logits[0, t] = float("-inf")

    def is_accepting(self):
        return True


NEG = float("-inf")


def test_mask_content():
    mask = _RecognizerMask(FakeRecognizer(), "L")
    assert [float(x) for x in mask([], RV)] == [0.0, NEG, 0.0, NEG, 0.0]
    assert [float(x) for x in mask([0], RV)] == [NEG, 0.0, NEG, 0.0, NEG]


def test_construction_rewinds_a_shared_recognizer():
    rec = FakeRecognizer()
    rec.try_advance_token_ids(torch.tensor([0, 1]))  # someone else left it mid-path
    mask = _RecognizerMask(rec, "L")
    assert rec.path == [] and rec.resets == 1
    assert [float(x) for x in mask([], RV)] == [0.0, NEG, 0.0, NEG, 0.0]


def test_extending_the_path_does_not_reset_but_diverging_does():
    rec = FakeRecognizer()
    mask = _RecognizerMask(rec, "L")
    base = rec.resets
    for prefix in ([], [0], [0, 1], [0, 1, 2]):
        mask(prefix, RV)
    assert rec.resets == base and rec.advanced == 3
    mask([0, 3], RV)  # diverges at position 1
    assert rec.resets == base + 1 and rec.path == [0, 3]


def test_returning_to_a_shorter_prefix_resets():
    rec = FakeRecognizer()
    mask = _RecognizerMask(rec, "L")
    mask([0, 1, 2], RV)
    before = rec.resets
    mask([0], RV)
    assert rec.resets == before + 1 and rec.path == [0]


def test_width_mismatch_between_recognizer_and_tokenizer():
    mask = _RecognizerMask(FakeRecognizer(), "L")
    wide = mask([], RV + 3)
    assert len(wide) == RV + 3 and all(float(x) == NEG for x in wide[RV:])
    assert len(mask([], RV - 2)) == RV - 2


def test_a_prefix_the_grammar_rejects_raises():
    with pytest.raises(ValueError, match="should be impossible"):
        _RecognizerMask(FakeRecognizer(), "L")([1], RV)


# =============================================================================================
# Leaf tolerance: float32 rounding below the target is not a violation of condition (i)
# =============================================================================================

class _Root(rt.NodeRT):
    def __init__(self, lp):
        self.lp = lp

    def log_weight(self):
        return 0.0

    def log_next(self):
        return self.lp


def _leaf(monkeypatch, gap):
    """An EnvelopeRT on a dominating mixture whose true value exceeds the envelope by ``gap``."""
    V, tok = 64, 7
    env = torch.log_softmax(torch.randn(V), dim=-1)
    true = env.clone()
    true[tok] += gap
    monkeypatch.setattr(rt, "_true_next", lambda root: true)
    state = rt.EnvelopeRT(root=_Root(env), vocab_size=V, eos_token_id=tok, models=[],
                          counters={}, leaf_scorers=(), context=(), dominating=True)
    return state, tok


def test_rounding_sized_leaf_gap_accepts(monkeypatch):
    for gap in (3.8e-6, 1.5e-5, 3.1e-5, 5e-4):
        state, tok = _leaf(monkeypatch, gap)
        assert state.log_leaf_acceptance(tok) == 0.0


def test_real_leaf_violation_still_raises(monkeypatch):
    state, tok = _leaf(monkeypatch, 0.05)
    with pytest.raises(ValueError, match="condition \\(i\\)"):
        state.log_leaf_acceptance(tok)


def test_envelope_above_target_pays_the_gap(monkeypatch):
    state, tok = _leaf(monkeypatch, -0.7)
    assert math.isclose(state.log_leaf_acceptance(tok), -0.7, rel_tol=1e-5)
