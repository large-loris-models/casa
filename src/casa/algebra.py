"""An algebra of envelopes.

MARS samples exactly from any target that admits an *envelope*: a function on prefixes that is
exact on complete sequences and never promises less for a prefix than for its children combined,

    (i)  E(w$) = phi(w$)                     for every complete sequence
    (ii) E(u) >= sum_a E(ua)                 for every prefix

This module lets you *write* a target as an expression and get an envelope for it, or a refusal
explaining that no envelope exists. Expressions are built with operators that mirror the maths:

    P, R = Model(llm_a), Model(llm_b)

    P * R                 product of experts          E = E_p * E_r
    P ** 0.5 * R ** 0.5   geometric mean, intersect   E = sqrt(E_p * E_r)
    P & R                 agreement, minimum          E = min(E_p, E_r)
    P | R                 union, mixture              E = (E_p + E_r) / 2
    P ** 2                sharpening                  E = E_p ** 2
    P * mask              hard constraint             E = E_p * 1[u in prefix(L)]
    mean([P, R], tau)     generalized mean            the whole family at once

Building an expression never checks anything. Validation happens once, at the root, in
``envelope()``. Two operations are refused there because no envelope exists for them at all, and
no exact sampler with next-token access can handle them without exponentially many model calls:

    P ** 0.5              tempering, on its own       refused
    P / R                 contrasting                 refused

Tempering is refused only *on its own*. Inside a product whose exponents sum to at least one it is
perfectly fine, because the product is what restores condition (ii). That is why a fractional power
does not raise when you write it: the expression tree is normalized first, and only the collected
exponents decide. ``(P ** 0.5) * (R ** 0.5)`` is an envelope; ``P ** 0.5`` alone is not.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------------------------


class NoEnvelope(Exception):
    """Raised when an expression admits no envelope.

    This is not a limitation of the implementation. For tempering and contrasting it is a theorem:
    every exact sampler with next-token access needs exponentially many model calls.
    """


@dataclass(frozen=True)
class Verdict:
    """The outcome of validating an expression.

    Attributes:
        ok: Whether an envelope exists.
        reason: Human-readable justification, citing the closure item or the obstruction.
        leaf_acceptance: If an envelope exists only with a final acceptance step, the guaranteed
            lower bound on that acceptance probability. ``None`` when no leaf step is needed.
    """

    ok: bool
    reason: str
    leaf_acceptance: Optional[float] = None

    def raise_if_bad(self) -> "Verdict":
        if not self.ok:
            raise NoEnvelope(self.reason)
        return self


# --------------------------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------------------------


class Potential(ABC):
    """A node in an envelope expression.

    A ``Potential`` is syntax, not a promise: it may or may not admit an envelope. Ask
    ``.verdict()`` for the answer and ``.envelope()`` for the thing MARS can actually sample.
    """

    # -- algebra ------------------------------------------------------------------------------

    def __mul__(self, other: "Potential | Scorer") -> "Potential":
        return Product.of(self, _coerce(other))

    def __rmul__(self, other: "Potential | Scorer") -> "Potential":
        return Product.of(_coerce(other), self)

    def __pow__(self, gamma: float) -> "Potential":
        if gamma <= 0:
            raise ValueError(f"exponent must be positive, got {gamma}")
        return Power(self, float(gamma))

    def __and__(self, other: "Potential") -> "Potential":
        """Agreement. ``P & R`` is the minimum, the generalized mean at tau = -infinity.

        The minimum is associative, so chaining is safe here, unlike ``|``.
        """
        return Mean.of([self, _coerce(other)], tau=-math.inf)

    def __or__(self, other: "Potential") -> "Potential":
        """Union. ``P | R`` is the equal-weight mixture, the generalized mean at tau = 1.

        Binary and left-associative, so ``P | R | S`` weighs them (0.25, 0.25, 0.5), not equally.
        For a uniform mixture of three or more, write ``union(P, R, S)``.
        """
        return Mean.of([self, _coerce(other)], tau=1.0)

    def __truediv__(self, other: "Potential") -> "Potential":
        raise NoEnvelope(
            "contrasting (a quotient of models) admits no envelope: the ratio is unbounded "
            "wherever the denominator is small, so condition (ii) fails and the target need not "
            "be normalizable. No exact sampler with next-token access can do better."
        )

    # -- validation ---------------------------------------------------------------------------

    @abstractmethod
    def verdict(self) -> Verdict:
        """Decide whether this expression admits an envelope, and say why."""

    def envelope(self) -> "Envelope":
        """Validate and wrap as an envelope MARS can sample. Raises ``NoEnvelope`` if it cannot."""
        v = self.verdict().raise_if_bad()
        return Envelope(self, v)

    @abstractmethod
    def leaves(self) -> List["Model"]:
        """Every base model appearing in this expression, in order, with duplicates."""


def _raise_scorer(sc: "Scorer", gamma: float) -> "Scorer":
    """``g ** gamma`` is itself a scorer, so an exponent passes straight through it.

    For ``gamma > 0`` the power of a prefix-monotone scorer is prefix-monotone and still lands in
    [0, 1], which is closure item (3) again. An indicator is unchanged by it, so a constrained
    model can be sharpened without leaving the algebra.
    """
    if gamma == 1.0:
        return sc
    fn, mask = sc.fn, sc.log_mask
    return Scorer(
        fn=lambda ctx, _f=fn, _g=gamma: _f(ctx) ** _g,
        name=f"{sc.name}**{gamma:g}",
        prefix_monotone=sc.prefix_monotone,
        log_mask=None if mask is None
        else (lambda ctx, v, _m=mask, _g=gamma: _scale(_m(ctx, v), _g)),
    )


def _scale(vec, gamma: float):
    """Multiply a log-mask by an exponent, whatever vector type it is."""
    try:
        return vec * gamma
    except TypeError:
        return [x * gamma for x in vec]


def _combine_leaf(values: Sequence[Optional[float]]) -> Optional[float]:
    """Fold the leaf-acceptance bounds of an expression's parts into one for the whole.

    ``None`` means a part needs no acceptance step. If every part is ``None`` so is the result.
    Otherwise the bounds multiply, since each level that reaches its target through a dominating
    envelope contributes its own factor to the final acceptance probability. Treat the number as a
    guide once operations are nested; the boolean, which decides whether MARS performs the step at
    all, is what has to be right.
    """
    present = [v for v in values if v is not None]
    if not present:
        return None
    out = 1.0
    for v in present:
        out *= v
    return out


def _coerce(x) -> Potential:
    if isinstance(x, Potential):
        return x
    raise TypeError(f"expected a Potential, got {type(x).__name__}")


@dataclass(frozen=True)
class Model(Potential):
    """A base potential: a language model, or any sub-stochastic next-token weighting.

    The prefix weight of a sub-stochastic potential is an envelope for it, since condition (ii)
    reads ``p(u) >= p(u) * sum_a p(a|u)``. For a language model this holds with equality.

    Attributes:
        llm: The CASA ``LLM`` this wraps.
        name: Label used in diagnostics and in the canonical form.
        prompt: Optional per-model prompt. Two ``Model`` nodes over the same ``llm`` with different
            prompts are the within-model ensemble setting: one model, two conditionings.
        sub_stochastic: Whether the weights at a prefix are guaranteed to sum to at most one.
            True for language models. A potential that is not sub-stochastic can still be used,
            but only under an exponent of at least one.
    """

    llm: object
    name: str = "p"
    prompt: Optional[str] = None
    sub_stochastic: bool = True

    def verdict(self) -> Verdict:
        return Verdict(True, f"{self.name}: prefix weight of a sub-stochastic potential")

    def leaves(self) -> List["Model"]:
        return [self]


@dataclass(frozen=True)
class Power(Potential):
    """``base ** gamma``.

    An envelope on its own only when ``gamma >= 1`` (sharpening). A fractional exponent is
    tempering, which has no envelope alone, but survives inside a product whose exponents sum to
    at least one. Normalization collects exponents before anything is decided, so this node never
    refuses by itself.
    """

    base: Potential
    gamma: float

    def verdict(self) -> Verdict:
        return Product.of(self).verdict()

    def leaves(self) -> List[Model]:
        return self.base.leaves()


@dataclass(frozen=True)
class Scorer(Potential):
    """A weighting of sequences by values in [0, 1].

    Two kinds, distinguished by ``prefix_monotone``:

    * prefix-monotone (``g(ua) <= g(u)``): an envelope factor outright, closure item (3). Hard
      constraints are the special case where ``g`` is an indicator of the prefix language.
    * output-only: reveals its weight only at complete sequences. MARS still samples exactly, by
      running on the envelope that ignores the scorer and accepting a yielded sequence with
      probability equal to its score. That is leaf rejection.

    Attributes:
        fn: Maps a token id sequence to a weight in [0, 1]. For output-only scorers it is called
            once per yielded sequence, on the generated tokens without the end-of-sequence marker,
            and its value is the acceptance probability.
        name: Label used in diagnostics.
        prefix_monotone: Whether ``fn`` is non-increasing along prefixes.
        log_mask: For prefix-monotone scorers, ``log_mask(context, vocab_size)`` returning
            ``log g(ua)`` for every token as a vector. Required, because scoring a whole
            vocabulary one Python call at a time would be silently ruinous; a grammar recognizer
            supplies it cheaply as a token bitmask.
    """

    fn: Callable[[Sequence[int]], float]
    name: str = "g"
    prefix_monotone: bool = True
    log_mask: Optional[Callable[[Sequence[int], int], object]] = None

    def verdict(self) -> Verdict:
        if self.prefix_monotone:
            return Verdict(True, f"{self.name}: prefix-monotone scorer, closure item (3)")
        return Verdict(True, f"{self.name}: output-only scorer, sampled with leaf rejection", 0.0)

    def leaves(self) -> List[Model]:
        return []


@dataclass(frozen=True)
class Product(Potential):
    """A weighted product ``prod_k base_k ** gamma_k``.

    An envelope when every exponent is positive and they sum to at least one, which is the
    generalized Hoelder inequality when they sum to exactly one and reduces to it otherwise.
    Prefix-monotone scorer factors ride along freely and do not count toward the exponent sum.

    Build these with ``*`` and ``**`` rather than directly; ``of`` flattens and collects so that
    ``(P ** 0.5) * (P ** 0.7)`` is recognized as ``P ** 1.2`` and therefore an envelope.
    """

    terms: Tuple[Tuple[Potential, float], ...]
    scorers: Tuple[Scorer, ...] = ()

    @staticmethod
    def of(*factors: Potential) -> "Product":
        terms: List[Tuple[Potential, float]] = []
        scorers: List[Scorer] = []

        def absorb(node: Potential, gamma: float) -> None:
            if isinstance(node, Power):
                absorb(node.base, gamma * node.gamma)
            elif isinstance(node, Product):
                for sub, g in node.terms:
                    absorb(sub, gamma * g)
                for sc in node.scorers:
                    scorers.append(_raise_scorer(sc, gamma))
            elif isinstance(node, Scorer):
                scorers.append(_raise_scorer(node, gamma))
            else:
                terms.append((node, gamma))

        for f in factors:
            absorb(f, 1.0)

        # Collect repeated bases so that P**0.5 * P**0.7 becomes P**1.2.
        collected: Dict[int, Tuple[Potential, float]] = {}
        order: List[int] = []
        for node, gamma in terms:
            # Merge by identity only. Merging two distinct but identical Model nodes would drop
            # one of them from `leaves()`, and a per-model prompt keyed on the dropped node would
            # then be silently ignored. Not merging them costs a forward pass and changes nothing
            # about the result.
            key = id(node)
            if key in collected:
                base, g = collected[key]
                collected[key] = (base, g + gamma)
            else:
                collected[key] = (node, gamma)
                order.append(key)
        return Product(tuple(collected[k] for k in order), tuple(scorers))

    @property
    def exponent_sum(self) -> float:
        return sum(g for _, g in self.terms)

    def verdict(self) -> Verdict:
        if not self.terms:
            return Verdict(
                False,
                "a target must contain at least one model; a scorer on its own has no next-token "
                "weights to sample from",
            )

        for base, gamma in self.terms:
            if gamma <= 0:
                return Verdict(
                    False,
                    f"exponent {gamma} on {_label(base)} is not positive; envelopes are not closed "
                    "under quotients",
                )
            sub = base.verdict()
            if not sub.ok:
                return sub

        total = self.exponent_sum
        if total < 1.0 - 1e-12:
            offenders = ", ".join(f"{_label(b)}**{g:g}" for b, g in self.terms)
            return Verdict(
                False,
                f"exponents sum to {total:g} < 1 ({offenders}). This is tempering: "
                "sum_a p(a|u)**gamma can exceed 1, so condition (ii) fails and no envelope of this "
                "form exists. Raise an exponent, or intersect with another model so the exponents "
                "sum to at least one.",
            )

        # A potential that is not sub-stochastic only behaves under an exponent of at least one.
        for base, gamma in self.terms:
            if isinstance(base, Model) and not base.sub_stochastic and gamma < 1.0 - 1e-12:
                return Verdict(
                    False,
                    f"{base.name} is not sub-stochastic, so it needs an exponent of at least 1; "
                    f"got {gamma:g}",
                )

        # A leaf requirement anywhere below must reach the root. If it does not, the sampler is
        # told no acceptance step is needed and quietly returns a biased distribution.
        leaf = _combine_leaf([b.verdict().leaf_acceptance for b, _ in self.terms]
                             + [sc.verdict().leaf_acceptance for sc in self.scorers])
        how = "Hoelder" if abs(total - 1.0) < 1e-12 else f"Hoelder after scaling (exponents sum to {total:g})"
        extra = f" with {len(self.scorers)} scorer factor(s)" if self.scorers else ""
        return Verdict(True, f"weighted product, closure item (1) by {how}{extra}", leaf)

    def leaves(self) -> List[Model]:
        out: List[Model] = []
        for base, _ in self.terms:
            out.extend(base.leaves())
        return out


@dataclass(frozen=True)
class Mean(Potential):
    """The generalized mean ``M_tau`` of several potentials, with weights in the simplex.

    This is the family Chan et al. (Ensembling Language Models with Sequential Monte Carlo,
    arXiv:2603.05432) characterize as the unique minimizers of weighted sums of alpha-divergences,
    with ``tau = 1 - alpha``. The whole family is here:

        tau -> -inf   minimum        agreement
        tau = -1      harmonic       agreement
        tau -> 0      product        agreement, and the same thing as a weighted product
        tau = 1       mixture        coverage
        tau = 2       quadratic      coverage
        tau -> +inf   maximum        coverage

    For ``tau <= 1`` the mean is concave and positively homogeneous, hence superadditive, so the
    mean of envelopes is an envelope and MARS samples exactly with no leaf step. For ``tau > 1`` it
    is subadditive and that fails; the mixture ``c * sum_k w_k E_k`` with ``c = 1 / min_k w_k``
    dominates the target at complete sequences instead, so MARS samples exactly with a leaf
    acceptance of at least ``1 / c``.
    """

    terms: Tuple[Potential, ...]
    weights: Tuple[float, ...]
    tau: float

    @staticmethod
    def of(terms: Sequence[Potential], tau: float, weights: Optional[Sequence[float]] = None) -> Potential:
        terms = tuple(terms)
        if len(terms) < 1:
            raise ValueError("a mean needs at least one term")
        if weights is None:
            weights = tuple(1.0 / len(terms) for _ in terms)
        else:
            weights = tuple(float(w) for w in weights)
            if len(weights) != len(terms):
                raise ValueError("weights and terms must have the same length")
            if any(w <= 0 for w in weights):
                # A zero weight makes c = 1 / min_k w_k infinite, so the coverage regime would have
                # no leaf-acceptance bound, and in log space it produces -inf + inf at any token the
                # zero-weighted model rules out. Drop the term instead of weighting it zero.
                raise ValueError(f"weights must be strictly positive, got {weights}")
            total = sum(weights)
            if not math.isclose(total, 1.0, rel_tol=1e-9):
                raise ValueError(f"weights must sum to 1, got {total}")
        # tau = 0 is the weighted geometric mean, which the product node already handles exactly.
        if tau == 0.0:
            return Product.of(*[t ** w for t, w in zip(terms, weights)])
        return Mean(terms, weights, float(tau))

    @property
    def needs_leaf_rejection(self) -> bool:
        return self.tau > 1.0

    @property
    def leaf_acceptance(self) -> float:
        """Guaranteed lower bound on the leaf acceptance probability, ``1 / c``."""
        return min(self.weights)

    def verdict(self) -> Verdict:
        inner = []
        for t in self.terms:
            sub = t.verdict()
            if not sub.ok:
                return sub
            inner.append(sub.leaf_acceptance)
        if self.tau <= 1.0:
            below = _combine_leaf(inner)
            note = "" if below is None else "; a term below it needs a leaf acceptance step"
            return Verdict(
                True,
                f"generalized mean at tau={_tau(self.tau)} is concave and positively homogeneous, "
                f"hence superadditive, so the mean of envelopes is an envelope{note}",
                below,
            )
        return Verdict(
            True,
            f"generalized mean at tau={_tau(self.tau)} is subadditive, so the mean of envelopes is "
            f"not one; MARS runs on the dominating mixture with leaf acceptance at least "
            f"{self.leaf_acceptance:g}",
            _combine_leaf(inner + [self.leaf_acceptance]),
        )

    def leaves(self) -> List[Model]:
        out: List[Model] = []
        for t in self.terms:
            out.extend(t.leaves())
        return out


# --------------------------------------------------------------------------------------------
# The validated result
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Envelope:
    """A target that MARS can sample exactly, together with why it can.

    Obtained from ``Potential.envelope()``. Holding one of these is a certificate: the expression
    was normalized and checked against the closure proposition, so an envelope exists.
    """

    expr: Potential
    verdict: Verdict

    @property
    def needs_leaf_rejection(self) -> bool:
        return self.verdict.leaf_acceptance is not None

    @property
    def leaf_acceptance(self) -> Optional[float]:
        return self.verdict.leaf_acceptance

    def models(self) -> List[Model]:
        return self.expr.leaves()

    def explain(self) -> str:
        lines = [f"target: {_label(self.expr)}", f"envelope: {self.verdict.reason}"]
        if self.needs_leaf_rejection:
            lines.append(f"leaf acceptance: at least {self.leaf_acceptance:g}")
        seen = {id(m.llm) for m in self.models()}
        lines.append(f"models: {len(self.models())} factor(s) over {len(seen)} distinct LM(s)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.explain()


# --------------------------------------------------------------------------------------------
# Sugar
# --------------------------------------------------------------------------------------------


def mean(terms: Sequence[Potential], tau: float, weights: Optional[Sequence[float]] = None) -> Potential:
    """The generalized mean of ``terms`` at power ``tau``. See :class:`Mean`."""
    return Mean.of(terms, tau, weights)


def intersect(*terms: Potential, weights: Optional[Sequence[float]] = None) -> Potential:
    """Weighted geometric mean, the product-of-experts target. ``tau = 0``."""
    if not terms:
        raise ValueError("intersect needs at least one potential")
    if weights is None:
        weights = [1.0 / len(terms)] * len(terms)
    return Product.of(*[t ** w for t, w in zip(terms, weights)])


def agree(*terms: Potential) -> Potential:
    """Minimum, the strictest consensus. ``tau = -infinity``."""
    return Mean.of(terms, tau=-math.inf)


def union(*terms: Potential, weights: Optional[Sequence[float]] = None) -> Potential:
    """Mixture. ``tau = 1``."""
    return Mean.of(terms, tau=1.0, weights=weights)


def constrain(p: Potential, mask, name: str = "L") -> Potential:
    """Restrict ``p`` to a prefix-closed language. This is exactly the CARS target.

    Args:
        p: The potential to constrain.
        mask: A callable ``mask(context, vocab_size)`` returning ``log g(ua)`` for every token,
            zero on tokens the language allows and negative infinity on the rest. Build one from a
            CASA grammar with :func:`casa.utils.envelope_runtime.grammar_mask`.
        name: Label for diagnostics.
    """
    return p * Scorer(fn=lambda ctx: 1.0, name=name, prefix_monotone=True, log_mask=mask)


def reweight(p: Potential, fn: Callable[[Sequence[int]], float], name: str = "g",
             prefix_monotone: bool = False) -> Potential:
    """Reweight ``p`` by a verifier or reward model. See :class:`Scorer`."""
    return p * Scorer(fn=fn, name=name, prefix_monotone=prefix_monotone)


def sharpen(p: Potential, gamma: float) -> Potential:
    """Raise ``p`` to a power. Envelopes exist only for ``gamma >= 1``; below that it is tempering."""
    return p ** gamma


# --------------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------------


def _tau(t: float) -> str:
    if t == -math.inf:
        return "-inf"
    if t == math.inf:
        return "+inf"
    return f"{t:g}"


def _label(node: Potential) -> str:
    if isinstance(node, Model):
        return node.name if node.prompt is None else f"{node.name}[{_short(node.prompt)}]"
    if isinstance(node, Scorer):
        return node.name
    if isinstance(node, Power):
        return f"{_label(node.base)}**{node.gamma:g}"
    if isinstance(node, Product):
        parts = [f"{_label(b)}**{g:g}" if g != 1.0 else _label(b) for b, g in node.terms]
        parts += [_label(s) for s in node.scorers]
        return " * ".join(parts) if parts else "1"
    if isinstance(node, Mean):
        inner = ", ".join(_label(t) for t in node.terms)
        if node.tau == -math.inf:
            return f"min({inner})"
        if node.tau == math.inf:
            return f"max({inner})"
        if node.tau == 1.0:
            return f"mix({inner})"
        return f"M[tau={_tau(node.tau)}]({inner})"
    return node.__class__.__name__.lower()


def _short(s: str, n: int = 18) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"
