"""Evaluating an envelope expression on a prefix.

:mod:`casa.algebra` decides *whether* a target admits an envelope. This module computes it. The two
are kept apart so that the algebra stays free of torch and of any model.

The runtime mirrors the expression one node at a time. Every node answers two questions about the
prefix it currently sits on:

    log_weight()   log E(u)              a scalar
    log_next()     log E(ua) for all a   a vector over the vocabulary

and can be advanced by one token. Because every node answers in *absolute* log-weight rather than
in conditionals, operations like the minimum compose correctly: ``min(E_p, E_r)`` at a prefix is not
determined by the two next-token distributions alone, it needs the two prefix weights as well.

All models in one expression must share a tokenizer. That is checked when the runtime is built.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from casa.algebra import Envelope, Mean, Model, Potential, Power, Product, Scorer

NEG_INF = float("-inf")
# Tolerance for the envelope conditions in float32. Summing a float32 log-softmax over a 128k
# vocabulary is off by around 1e-5, and the difference of two such values at a leaf by about
# the same, so a strict comparison would report violations that are rounding. One part in a
# thousand is far above rounding and far below any real violation; the sampler uses the
# same value at expansion.
LOG_SLACK = math.log1p(1e-3)


class TokenizerMismatch(Exception):
    """Raised when an expression mixes models that do not share a token space.

    MARS evaluates every model on the same token context, so a shared vocabulary is required. Two
    models with different tokenizers can still be ensembled, but only after both are mapped to a
    common byte alphabet, which this runtime does not do.
    """


# --------------------------------------------------------------------------------------------
# Node runtimes
# --------------------------------------------------------------------------------------------


class NodeRT:
    """Runtime for one expression node, positioned at some prefix."""

    def log_weight(self) -> float:
        raise NotImplementedError

    def log_next(self) -> torch.Tensor:
        raise NotImplementedError

    def advance(self, token: int) -> "NodeRT":
        raise NotImplementedError


class ModelRT(NodeRT):
    """A base language model at some prefix, with its own prompt.

    Two ``ModelRT`` over the same ``LLM`` with different prompts is the within-model ensemble: one
    set of weights, two conditionings.

    Advancing is free. A prefix costs one forward pass, taken lazily the first time somebody asks
    for a weight, and that single pass yields every conditional along the prefix at once because
    the logits at position ``j`` predict token ``j+1``. This is what makes a revisited trie node
    cost nothing: MARS walks down through cached nodes without asking any model anything, and only
    a node it has never expanded triggers a pass.
    """

    def __init__(self, node: Model, prompt_ids: Sequence[int], device, vocab_size: int,
                 temperature: float, counter: Dict[str, int]):
        if not prompt_ids:
            raise ValueError(
                f"model {node.name!r} has an empty prompt. A forward pass yields the conditional "
                "for a token only from the position before it, so the first generated token would "
                "have none. Give the model a prompt, or at least a beginning-of-sequence token."
            )
        self.node = node
        self.llm = node.llm
        self.prompt_ids = list(prompt_ids)
        self.device = device
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.context: List[int] = []
        self._counter = counter
        self._log_weight: Optional[float] = None
        self._log_cond: Optional[torch.Tensor] = None

    def _shape(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] > self.vocab_size:
            logits = logits[..., : self.vocab_size]
        elif logits.shape[-1] < self.vocab_size:
            pad = torch.full(logits.shape[:-1] + (self.vocab_size - logits.shape[-1],),
                             NEG_INF, device=logits.device)
            logits = torch.cat([logits, pad], dim=-1)
        # A language model's logits become a distribution; a general potential's are already
        # log-weights and must not be renormalized, or a 0/1 mask would come back as a uniform
        # choice among the tokens it permits. Temperature applies only to the former: rescaling an
        # unnormalized potential would change its weights, and could lift their sum above one.
        if not self.node.sub_stochastic:
            return logits
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return torch.log_softmax(logits, dim=-1)

    def _materialize(self) -> None:
        if self._log_cond is not None:
            return
        ids = self.prompt_ids + self.context
        with torch.no_grad():
            out = self.llm.model(torch.tensor([ids], device=self.device), use_cache=False)
        self._counter["passes"] += 1
        lp = self._shape(out.logits[0].float())
        if self.context:
            # Position len(prompt) - 1 + i predicts context[i]. Gathered in one go: indexing a
            # device tensor once per token costs a synchronization each time.
            base = len(self.prompt_ids) - 1
            idx = torch.tensor(self.context, device=lp.device).unsqueeze(1)
            rows = lp[base: base + len(self.context)]
            self._log_weight = float(rows.gather(1, idx).sum())
        else:
            self._log_weight = 0.0
        # Clone, do not keep a view. `lp` is the full [sequence, vocabulary] logits, and a row of
        # it retains the entire storage: 88 MiB per prefix at a 128k vocabulary instead of 0.5 MiB.
        # Held across a few hundred cached prefixes that is tens of gigabytes, which is exactly how
        # an 8B model filled a 44 GiB card.
        self._log_cond = lp[-1].clone()

    def log_weight(self) -> float:
        if not self.context:
            return 0.0  # the empty prefix weighs one, and no model needs to be asked
        if self._log_weight is None:
            self._materialize()
        return self._log_weight

    def log_next(self) -> torch.Tensor:
        self._materialize()
        return self._log_cond + self.log_weight()

    def advance(self, token: int) -> "ModelRT":
        child = ModelRT.__new__(ModelRT)
        # Shared: everything describing the model and the run. The counter is shared deliberately,
        # so one model's calls are counted once however far the descent goes.
        child.node = self.node
        child.llm = self.llm
        child.prompt_ids = self.prompt_ids
        child.device = self.device
        child.vocab_size = self.vocab_size
        child.temperature = self.temperature
        child._counter = self._counter
        # Per-prefix, and never inherited.
        child.context = self.context + [token]
        child._log_weight = None
        child._log_cond = None
        return child



class PowerRT(NodeRT):
    def __init__(self, child: NodeRT, gamma: float):
        self.child, self.gamma = child, gamma

    def log_weight(self) -> float:
        return self.gamma * self.child.log_weight()

    def log_next(self) -> torch.Tensor:
        return self.gamma * self.child.log_next()

    def advance(self, token: int) -> "PowerRT":
        return PowerRT(self.child.advance(token), self.gamma)



class ProductRT(NodeRT):
    """``prod_k E_k ** gamma_k``, times any prefix-monotone scorer factors."""

    def __init__(self, terms: List[Tuple[NodeRT, float]], scorers: List["ScorerRT"]):
        self.terms, self.scorers = terms, scorers

    def log_weight(self) -> float:
        w = sum(g * c.log_weight() for c, g in self.terms)
        return w + sum(s.log_weight() for s in self.scorers)

    def log_next(self) -> torch.Tensor:
        if not self.terms:
            raise ValueError("a product with no model factor has no next-token weights")
        out = None
        for c, g in self.terms:
            v = c.log_next() * g
            out = v if out is None else out + v
        for s in self.scorers:
            out = out + s.log_next_mask(out.shape[-1], out.device)
        return out

    def advance(self, token: int) -> "ProductRT":
        return ProductRT(
            [(c.advance(token), g) for c, g in self.terms],
            [s.advance(token) for s in self.scorers],
        )



class MeanRT(NodeRT):
    """The generalized mean at power ``tau``, in log space.

        log M_tau = (1 / tau) * logsumexp_k ( log w_k + tau * log E_k )

    with the limits handled directly: ``tau -> -inf`` is the minimum, ``tau -> +inf`` the maximum,
    and ``tau = 0`` never reaches here because the algebra rewrites it as a weighted product.
    """

    def __init__(self, children: List[NodeRT], weights: Sequence[float], tau: float):
        self.children = children
        self.weights = list(weights)
        self.tau = tau
        if any(w <= 0 for w in self.weights):
            raise ValueError(
                f"generalized-mean weights must be strictly positive, got {self.weights}; a zero "
                "weight gives -inf + inf for negative tau, which is NaN"
            )
        self._logw = [math.log(w) for w in self.weights]

    def _combine_scalar(self, vals: List[float]) -> float:
        if self.tau == -math.inf:
            return min(vals)
        if self.tau == math.inf:
            return max(vals)
        t = torch.tensor([lw + self.tau * v for lw, v in zip(self._logw, vals)])
        return float(torch.logsumexp(t, dim=0)) / self.tau

    def _combine_vec(self, vecs: List[torch.Tensor]) -> torch.Tensor:
        stack = torch.stack(vecs, dim=0)
        if self.tau == -math.inf:
            return stack.min(dim=0).values
        if self.tau == math.inf:
            return stack.max(dim=0).values
        lw = torch.tensor(self._logw, device=stack.device).unsqueeze(1)
        return torch.logsumexp(lw + self.tau * stack, dim=0) / self.tau

    @property
    def subadditive(self) -> bool:
        """``tau > 1`` is the coverage regime, where the mean of envelopes is not an envelope."""
        return self.tau > 1.0

    def _dominating_vec(self, vecs: List[torch.Tensor]) -> torch.Tensor:
        """``c * sum_k w_k E_k`` with ``c = 1 / min_k w_k``, which dominates ``M_tau``."""
        stack = torch.stack(vecs, dim=0)
        lw = torch.tensor(self._logw, device=stack.device).unsqueeze(1)
        return torch.logsumexp(lw + stack, dim=0) - math.log(min(self.weights))

    def _dominating_scalar(self, vals: List[float]) -> float:
        t = torch.tensor([lw + v for lw, v in zip(self._logw, vals)])
        return float(torch.logsumexp(t, dim=0)) - math.log(min(self.weights))

    def log_weight(self) -> float:
        vals = [c.log_weight() for c in self.children]
        if self.subadditive:
            return self._dominating_scalar(vals)
        return self._combine_scalar(vals)

    def log_next(self) -> torch.Tensor:
        """The envelope. In the coverage regime this is the dominating mixture, not the mean."""
        vecs = [c.log_next() for c in self.children]
        if self.subadditive:
            return self._dominating_vec(vecs)
        return self._combine_vec(vecs)

    def log_next_true(self) -> torch.Tensor:
        """The target itself, which differs from the envelope only in the coverage regime."""
        return self._combine_vec([_true_next(c) for c in self.children])

    def advance(self, token: int) -> "MeanRT":
        return MeanRT([c.advance(token) for c in self.children], self.weights, self.tau)



class ScorerRT:
    """A prefix-monotone scorer factor.

    Needs the weight of every continuation, not just of the current prefix, so it must supply a
    vectorized mask. A grammar recognizer provides one cheaply as a token bitmask; a general
    Python scorer would have to be called once per vocabulary entry, which is refused rather than
    done silently.
    """

    def __init__(self, node: Scorer, context: Optional[List[int]] = None):
        self.node = node
        self.context = context or []
        if getattr(node, "log_mask", None) is None:
            raise NotImplementedError(
                f"scorer {node.name!r} is prefix-monotone but supplies no vectorized mask. "
                "Give it a `log_mask(context, vocab_size)` returning log g(ua) for every token, "
                "or mark it output-only so MARS uses leaf rejection instead."
            )

    def log_weight(self) -> float:
        v = self.node.fn(tuple(self.context))
        return math.log(v) if v > 0 else NEG_INF

    def log_next_mask(self, vocab_size: int, device) -> torch.Tensor:
        return self.node.log_mask(self.context, vocab_size).to(device)

    def advance(self, token: int) -> "ScorerRT":
        return ScorerRT(self.node, self.context + [token])


# --------------------------------------------------------------------------------------------
# Building a runtime from a validated envelope
# --------------------------------------------------------------------------------------------


@dataclass
class EnvelopeRT:
    """The root of an envelope runtime, positioned at the empty prefix."""

    root: NodeRT
    vocab_size: int
    eos_token_id: int
    models: List[ModelRT]          # the models at the *root*, for tokenizer access only
    counters: Dict[int, Dict[str, int]] = field(default_factory=dict)
    leaf_scorers: Tuple[Scorer, ...] = ()
    context: Tuple[int, ...] = ()
    dominating: bool = False       # does any subadditive mean make the envelope loose at leaves?

    def log_weight(self) -> float:
        return self.root.log_weight()

    def log_next(self) -> torch.Tensor:
        return self.root.log_next()

    def log_ratios(self) -> torch.Tensor:
        """``log E(ua) - log E(u)`` for every token.

        This is what a MARS trie node stores. Condition (ii) says it exponentiates to at most one
        in total, and the shortfall is exactly the rejection mass paid at this expansion.
        """
        w = self.root.log_weight()
        if w == NEG_INF:
            return torch.full((self.vocab_size,), NEG_INF)
        return self.root.log_next() - w

    def forced_stop_ratios(self) -> torch.Tensor:
        """Ratios for a prefix that has reached the length bound: stop, with probability one.

        The paper's model assumes the end-of-sequence marker is almost surely produced, and says
        this can be enforced with a maximum length. Enforcing it is what this is. Without it, a
        model that simply does not want to stop yet, which is every language model a few tokens
        into a sentence, sends every descent into the length bound to be discarded, and a run
        returns nothing at all after doing all the work.
        """
        out = torch.full((self.vocab_size,), NEG_INF)
        out[self.eos_token_id] = 0.0
        return out

    def log_truncation_acceptance(self) -> float:
        """``log phi(u) - log E(u)`` when the sequence is stopped at the bound.

        The analogue of :meth:`log_leaf_acceptance` for a forced stop, where the sequence's weight
        is the *prefix* weight rather than the weight of a prefix followed by the marker.
        """
        gap = 0.0
        for sc in self.leaf_scorers:
            v = sc.fn(self.context)
            if not 0.0 <= v <= 1.0:
                raise ValueError(
                    f"scorer {sc.name!r} returned {v}, which is outside [0, 1]; it cannot be used "
                    "as an acceptance probability"
                )
            gap += math.log(v) if v > 0 else NEG_INF
        if self.dominating:
            gap += _true_weight(self.root) - self.root.log_weight()
        return min(gap, 0.0)

    def log_leaf_acceptance(self, token: int) -> float:
        """``log phi(w$) - log E(w$)`` for a yielded complete sequence.

        Zero whenever the envelope is exact at leaves, which is every target except the coverage
        regime of the generalized mean. There MARS runs on a dominating mixture and pays the
        difference once, at the leaf, with acceptance bounded below by ``min_k w_k``.
        """
        # Output-only scorers reveal their weight only here, on the complete sequence, which is
        # exactly why they cost a leaf step rather than tightening the trie.
        gap = 0.0
        # Scorers see the generated tokens, without the end-of-sequence marker: a verifier scores
        # the text that was produced, and this is the same sequence `SamplingResult.text` decodes.
        for sc in self.leaf_scorers:
            v = sc.fn(self.context)
            if not 0.0 <= v <= 1.0:
                raise ValueError(
                    f"scorer {sc.name!r} returned {v}, which is outside [0, 1]; it cannot be used "
                    "as an acceptance probability"
                )
            gap += math.log(v) if v > 0 else NEG_INF
        if self.dominating:
            # Only a subadditive mean makes the envelope differ from the target at a leaf. Without
            # one this gap is identically zero, and computing it would force a forward pass per
            # yielded sequence for a number already known to be zero.
            true = _true_next(self.root)
            env = self.root.log_next()
            gap += float(true[token]) - float(env[token])
        if gap > LOG_SLACK:
            raise ValueError(
                f"the envelope is below the target at a complete sequence (by {gap:.9f} in log "
                "space), so it does not dominate and no acceptance probability exists. This is a "
                "violation of condition (i)."
            )
        # Within the slack the envelope equals the target up to rounding: accept with probability
        # one. A strict 1e-6 here failed 20 of 50 maximum-ensemble shards on gaps of 4e-6 to 3e-5.
        return min(gap, 0.0)

    def advance(self, token: int) -> "EnvelopeRT":
        return EnvelopeRT(self.root.advance(token), self.vocab_size, self.eos_token_id,
                          self.models, self.counters, self.leaf_scorers,
                          self.context + (token,), self.dominating)

    def forward_passes(self) -> int:
        """Total model calls so far, across every model in the expression."""
        return sum(c["passes"] for c in self.counters.values())


def _true_weight(node: NodeRT) -> float:
    """The target's prefix weight, which differs from the envelope's only under a subadditive mean."""
    if isinstance(node, MeanRT):
        return node._combine_scalar([_true_weight(c) for c in node.children])
    if isinstance(node, PowerRT):
        return node.gamma * _true_weight(node.child)
    if isinstance(node, ProductRT):
        w = sum(g * _true_weight(c) for c, g in node.terms)
        return w + sum(sc.log_weight() for sc in node.scorers)
    return node.log_weight()


def _true_next(node: NodeRT) -> torch.Tensor:
    """The target's next-token weights, which coincide with the envelope's except under a
    subadditive mean."""
    fn = getattr(node, "log_next_true", None)
    if fn is not None:
        return fn()
    if isinstance(node, PowerRT):
        return node.gamma * _true_next(node.child)
    if isinstance(node, ProductRT):
        out = None
        for c, g in node.terms:
            v = _true_next(c) * g
            out = v if out is None else out + v
        for s in node.scorers:
            out = out + s.log_next_mask(out.shape[-1], out.device)
        return out
    return node.log_next()


@dataclass
class Plan:
    """Everything about a target that does not change from one descent to the next.

    Validation and prompt tokenization happen once here. A descent then calls :meth:`fresh`, which
    only allocates node runtimes. Doing this per descent instead would mean comparing two
    hundred-thousand-entry vocabularies and re-applying a chat template every time a sample was
    rejected.
    """

    envelope: Envelope
    prompt_ids: Dict[int, List[int]]
    vocab_size: int
    eos_token_id: int
    device: object
    temperature: float
    _dominating: bool = False

    def fresh(self, counters: Optional[Dict[int, Dict[str, int]]] = None) -> EnvelopeRT:
        """A runtime at the empty prefix.

        Pass ``counters`` to share model-call accounting across several runtimes. Anything that
        builds a runtime per prefix otherwise counts each one separately and reports a total far
        below what was actually spent.
        """
        built: List[ModelRT] = []
        counters = {} if counters is None else counters
        leaf: List[Scorer] = []

        def go(node: Potential) -> NodeRT:
            if isinstance(node, Model):
                counter = counters.setdefault(id(node), {"passes": 0})
                rt = ModelRT(node, self.prompt_ids[id(node)], self.device, self.vocab_size,
                             self.temperature, counter)
                built.append(rt)
                return rt
            if isinstance(node, Power):
                return PowerRT(go(node.base), node.gamma)
            if isinstance(node, Product):
                prefix_scorers = []
                for sc in node.scorers:
                    if sc.prefix_monotone:
                        prefix_scorers.append(ScorerRT(sc))
                    else:
                        leaf.append(sc)
                return ProductRT([(go(b), g) for b, g in node.terms], prefix_scorers)
            if isinstance(node, Mean):
                return MeanRT([go(t) for t in node.terms], node.weights, node.tau)
            if isinstance(node, Scorer):
                raise ValueError("a scorer cannot be the whole target; multiply it into a model")
            raise TypeError(f"unhandled expression node {type(node).__name__}")

        root = go(self.envelope.expr)
        return EnvelopeRT(root, self.vocab_size, self.eos_token_id, built, counters, tuple(leaf),
                          (), self._dominating)


def plan(envelope: Envelope, prompts: Optional[Dict[int, str]] = None,
         temperature: float = 1.0) -> "Plan":
    """Prepare a validated envelope for sampling.

    Args:
        envelope: The result of ``Potential.envelope()``.
        prompts: Optional override mapping ``id(Model node)`` to a prompt string. Models carrying
            their own ``prompt`` use it; this is for supplying one task instance at a time.
        temperature: Sampling temperature applied to every model's logits.

    Returns:
        A :class:`Plan`. Call :meth:`Plan.fresh` for a runtime positioned at the empty prefix.

    Raises:
        TokenizerMismatch: If the expression mixes models with different token spaces.
    """
    models = envelope.models()
    if not models:
        raise ValueError("expression contains no model")

    ref = models[0].llm
    ref_tok = ref.tokenizer
    for m in models[1:]:
        if not _same_token_space(ref_tok, m.llm.tokenizer):
            raise TokenizerMismatch(
                f"{models[0].name} and {m.name} do not share a token space "
                f"({ref_tok.__class__.__name__}/{len(ref_tok)} vs "
                f"{m.llm.tokenizer.__class__.__name__}/{len(m.llm.tokenizer)}). "
                "MARS evaluates every model on the same token context, so all models in one "
                "expression must share a tokenizer. Map both to a byte alphabet first, or pair "
                "models from the same family."
            )

    prompt_ids: Dict[int, List[int]] = {}
    for m in models:
        text = (prompts or {}).get(id(m), m.prompt) or ""
        prompt_ids[id(m)] = m.llm.tokenizer.encode(m.llm.format_prompt(text),
                                                   add_special_tokens=False)

    def subadditive(node: Potential) -> bool:
        if isinstance(node, Mean):
            return node.tau > 1.0 or any(subadditive(t) for t in node.terms)
        if isinstance(node, Power):
            return subadditive(node.base)
        if isinstance(node, Product):
            return any(subadditive(b) for b, _ in node.terms)
        return False

    if ref_tok.eos_token_id is None:
        raise ValueError(
            "the tokenizer has no eos_token_id, so no sequence can ever be completed and every "
            "descent would run to the length bound and be discarded"
        )

    return Plan(envelope, prompt_ids, len(ref_tok), ref_tok.eos_token_id,
                ref.model.device, temperature, subadditive(envelope.expr))


def _same_token_space(a, b) -> bool:
    if a is b:
        return True
    if len(a) != len(b):
        return False
    va, vb = a.get_vocab(), b.get_vocab()
    return va == vb


# --------------------------------------------------------------------------------------------
# Grammar constraints
# --------------------------------------------------------------------------------------------


class _RecognizerMask:
    """Adapts a stateful grammar recognizer to the random-access queries MARS makes.

    CASA's recognizers are sequential: they consume tokens in order and remember how many they
    have seen. CARS can use them directly because a descent is a single forward walk and the
    recognizer is reset between attempts. MARS revisits prefixes in whatever order the trie sends
    it, so this wrapper tracks the path the recognizer currently stands on and replays from the
    start whenever the requested prefix diverges from it. Extending the current path, which is the
    common case inside one descent, costs only the new tokens.
    """

    def __init__(self, recognizer, name: str = "L"):
        self.rec = recognizer
        self.name = name
        self._path: List[int] = []
        # A CASA Grammar holds a single recognizer and shares it. Whoever used it last may have
        # left it part-way down a path, and an empty `_path` claims it stands at the root, so the
        # mask for the first token would silently be some other prefix's.
        self.rec.reset()

    def _sync(self, context: List[int]) -> None:
        shared = 0
        while (shared < len(self._path) and shared < len(context)
               and self._path[shared] == context[shared]):
            shared += 1
        if shared < len(self._path):
            self.rec.reset()
            self._path = []
        if len(context) > len(self._path):
            if not self.rec.try_advance_token_ids(torch.tensor(context)):
                # The incremental extension failed. Before concluding that the recognizer and
                # the sampler disagree, rule out the recognizer standing somewhere other than
                # where `_path` says: anything else that touched it, a validity check on a sample
                # or another user of the same grammar, moves it without updating `_path`. A clean
                # replay from the root is the ground truth for this prefix.
                self.rec.reset()
                if self.rec.try_advance_token_ids(torch.tensor(context)):
                    warnings.warn(
                        f"grammar {self.name!r}: the recognizer had drifted from the path the mask "
                        "tracked (it accepted the prefix on a clean replay but not incrementally); "
                        "recovered by replaying from the root."
                    )
                else:
                    consumed = getattr(self.rec, "current_index", None)
                    raise ValueError(
                        f"grammar {self.name!r} rejected a prefix MARS had already accepted, even "
                        f"on a clean replay from the root: it consumed {consumed} of "
                        f"{len(context)} tokens of {list(context)}. The recognizer and the sampler "
                        "have disagreed about validity, which should be impossible: the mask is "
                        "what chose every token on this path."
                    )
            self._path = list(context)

    def __call__(self, context: Sequence[int], vocab_size: int) -> torch.Tensor:
        self._sync(list(context))
        width = getattr(self.rec, "vocab_size", vocab_size)
        row = torch.zeros(1, width)
        self.rec.apply_token_bitmask(row, self.rec.filter_vocab())
        out = row[0]
        if width > vocab_size:
            return out[:vocab_size]
        if width < vocab_size:
            return torch.cat([out, torch.full((vocab_size - width,), NEG_INF)])
        return out


def grammar_mask(grammar, name: str = "L") -> _RecognizerMask:
    """A vectorized validity mask for a CASA :class:`~casa.grammar.Grammar`.

    Pass the result to :func:`casa.algebra.constrain`::

        from casa import Grammar
        from casa.algebra import Model, constrain
        from casa.utils.envelope_runtime import grammar_mask

        g = Grammar.from_string(spec, llm.tokenizer)
        target = constrain(Model(llm, name="P"), grammar_mask(g))

    Constraining a single model this way gives exactly the CARS target, which is the point: CARS
    is one expression in this algebra rather than a separate algorithm.
    """
    rec = getattr(grammar, "recognizer", grammar)
    return _RecognizerMask(rec, name)


def build(envelope: Envelope, prompts: Optional[Dict[int, str]] = None,
          temperature: float = 1.0) -> EnvelopeRT:
    """One runtime for a validated envelope, positioned at the empty prefix.

    Convenience for a single use. Anything that descends repeatedly should hold a :class:`Plan`
    and call :meth:`Plan.fresh`, so validation and tokenization are not repeated.
    """
    return plan(envelope, prompts, temperature).fresh()
