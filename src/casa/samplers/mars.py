"""Modular Adaptive Rejection Sampling.

MARS draws exact samples from any target that admits an envelope: combinations of language models
built with :mod:`casa.algebra`, including intersections, agreement and coverage ensembles,
sharpening, reweighting, and hard constraints.

It is CARS with one thing generalized. CARS stores, at each visited prefix, the model's next-token
distribution masked by grammar validity; the mask is 1 on valid tokens and 0 on invalid ones, and a
rejection sets an entry to 0 and propagates the shrunken mass to the parent. MARS stores the
*envelope ratio* ``log E(ua) - log E(u)`` instead of that masked distribution. Everything else, the
trie, the propagation, the telescoping argument, is unchanged, and a hard constraint makes the
ratio factor back into "model conditional times validity mask", which is CARS exactly.

Two consequences worth knowing. A rejection can now be partial: where CARS discovers an invalid
continuation by drawing it, MARS flips one coin at the moment a prefix is first expanded, with
survival probability equal to the total child mass over the parent's bound, and never draws the
token at all. And no rejection ever happens at a leaf, because the envelope is exact on complete
sequences, so once a descent reaches the end-of-sequence token the sample is accepted. Where the
envelope does not start out exact at leaves -- the coverage regime, and any output-only scorer --
MARS tightens the bound on the stop token to the target's own weight, which is known at the prefix
it completes, rather than leaving a gap to be flipped for at the leaf. Both orders are exact and
cost the same forward passes; only this one leaves a tightened bound behind, which is why the
non-adaptive modes (``adaptive=False``) keep the coin.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import torch

from casa.utils import envelope_runtime as rt
from casa.utils.draw import draw_log, draw_prepared, prepare_log
from casa.algebra import Envelope, Potential
from casa.samplers.base import SamplingResult
from casa.utils.helpers import print_progress
from casa.utils.oracle_trie import Trie

NEG_INF = float("-inf")

#: How far above one a prefix's child mass may sit before it counts as a real violation rather
#: than arithmetic. Summing a float32 log-softmax over a 150k vocabulary is off by around 5e-5 in
#: either direction, fifty times a 1e-6 tolerance, so a tight threshold rejects every ordinary
#: model on its first expansion. A genuine violation is a factor, not a fifth of a permille.
_LOG_SLACK = rt.LOG_SLACK  # one tolerance for both envelope conditions, defined with the runtime


class MARS:
    """Exact sampling from a combination of language models.

    Args:
        target: A :class:`casa.algebra.Envelope`, or a :class:`casa.algebra.Potential` which is
            validated on the spot. Passing a potential with no envelope raises ``NoEnvelope``,
            naming the obstruction.
        max_new_tokens: Length bound.
        verbose: Print a progress line per sample.
        temperature: Applied to every model's logits before the envelope is formed.
        on_max_length: What a prefix that reaches the bound does.

            ``"stop"`` (the default) ends it, as if the models had produced the end-of-sequence
            marker with probability one. The target is then the combination truncated to this
            length, which is what enforcing termination by a maximum length means.

            ``"discard"`` throws the descent away instead. That is also exact, for the target
            conditioned on terminating within the bound, but every language model a few tokens
            into a sentence wants to keep going, so nearly every descent is wasted and a short
            bound can return nothing at all after doing all the work.
        adaptive: Whether to keep what each rejection teaches.

            ``False`` starts every descent from the original envelope, discarding the tightened
            bounds. That is plain rejection sampling from the same envelope: same target, same
            exactness, same per-descent behaviour, but nothing is remembered, so the acceptance
            rate never moves. It is the reference point for what adaptivity buys, and the
            only difference between the two is whether the trie survives a descent.
        cache: With ``adaptive=False`` only: keep every expanded prefix's envelope ratios across
            descents, but never tighten a bound. That is rejection sampling with the same prefix
            cache MARS has, so a revisited prefix costs no forward pass, and the survival coin is
            flipped again from the stored ratios on every visit, as plain rejection sampling
            would flip it. Against MARS this isolates what the learned bounds buy from what the
            cache buys; on a target that never rejects the two are the same algorithm.
    """

    def __init__(self, target, max_new_tokens: int = 512, verbose: bool = False,
                 temperature: float = 1.0, on_max_length: str = "stop",
                 adaptive: bool = True, cache: bool = False):
        self.envelope: Envelope = target.envelope() if isinstance(target, Potential) else target
        if not isinstance(self.envelope, Envelope):
            raise TypeError(f"expected an Envelope or Potential, got {type(target).__name__}")
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.temperature = temperature
        if on_max_length not in ("stop", "discard"):
            raise ValueError(f"on_max_length must be 'stop' or 'discard', got {on_max_length!r}")
        self.on_max_length = on_max_length
        if cache and adaptive:
            raise ValueError("cache=True is rejection sampling with a prefix cache; MARS already "
                             "keeps its trie, so pass adaptive=False with it")
        self.adaptive = adaptive
        self.cache = cache
        self.trie = Trie()
        self.stats = _Stats()
        self._key = None

    def reset(self) -> None:
        """Forget every bound learned so far and start from the original envelope."""
        self.trie = Trie()
        self.stats = _Stats()

    @property
    def is_adaptive(self) -> bool:
        return self.adaptive

    @property
    def root_mass(self) -> float:
        """The bound currently held at the root.

        Starts at the envelope's own value and never increases. The probability that a descent
        yields a sample is the normalizer over this, so a falling root mass is a rising acceptance
        rate; that is the monotonicity claim, in the form you can actually watch.
        """
        root = self.trie.root
        if root.raw_logprob is None:
            return 1.0
        bounds = (root.raw_logprob[0] if root.log_theta is None
                  else root.raw_logprob[0] + root.log_theta[0])
        return float(torch.exp(torch.logsumexp(bounds, dim=0)))

    # -- public ------------------------------------------------------------------------------

    def sample(self, prompt: str = "", n_samples: int = 1, max_attempts: int = 100,
               prompts: Optional[Dict[int, str]] = None) -> List[SamplingResult]:
        """Draw ``n_samples`` exact samples.

        Args:
            prompt: Prompt for every model that does not carry its own.
            n_samples: How many samples to return.
            max_attempts: Give up on a sample after this many rejected descents. Samples that are
                given up on are omitted, and ``stats.timeouts`` records how often that happened.
            prompts: Optional per-model prompt override, keyed by ``id`` of the algebra's ``Model``
                node. Use this to run one task instance at a time without rebuilding the target.

        Returns:
            The successful samples, in order.
        """
        # Merge, do not replace. A caller overriding one model's prompt must not silently blank
        # every other model's, which is what taking `prompts` wholesale used to do.
        resolved = {**self._default_prompts(prompt), **(prompts or {})}

        # Validation and prompt tokenization happen once, in the plan. Each descent then gets a
        # fresh runtime, exactly as CARS rebuilds its cache per attempt: a descent branches and
        # key-value caches mutate in place, so reusing one across descents would hand the second
        # descent a cache holding the first one's tokens. Nothing is lost, because the trie caches
        # the envelope ratios themselves and a revisited prefix costs no forward pass at all.
        plan = rt.plan(self.envelope, prompts=resolved, temperature=self.temperature)
        fresh = plan.fresh

        # The trie belongs to the target, not to one call. Everything MARS learns about a prefix
        # stays valid for every later sample from the same target, and throwing it away between
        # calls would discard exactly the thing that makes the acceptance rate climb. It is reset
        # only when the conditioning changes, which makes the stored bounds meaningless.
        key = tuple(sorted(resolved.items()))
        if key != self._key:
            self.reset()
            self._key = key
        results: List[SamplingResult] = []

        for i in range(n_samples):
            attempts = 0
            got = None
            for _ in range(max_attempts):
                attempts += 1
                if not self.adaptive and not self.cache:
                    # Forget everything learned. Each descent then faces the original envelope,
                    # which is exactly plain rejection sampling.
                    self.trie = Trie()
                got = self._descend(fresh)
                if got is not None:
                    break
            if got is not None:
                got.attempts = attempts
                results.append(got)
                self.stats.accepted += 1
            else:
                self.stats.timeouts += 1
            print_progress(i + 1, n_samples, attempts, max_attempts, self.verbose,
                           timeout=got is None)
        return results

    def _default_prompts(self, prompt: str) -> Dict[int, str]:
        return {id(m): (m.prompt if m.prompt is not None else prompt)
                for m in self.envelope.models()}

    @property
    def acceptance_rate(self) -> float:
        """Fraction of descents so far that yielded a sample.

        A cumulative ratio, so it is not itself monotone; the quantity the theorem is about is
        :attr:`root_mass`, which never increases.
        """
        total = self.stats.descents
        return self.stats.accepted / total if total else 0.0

    # -- one descent -------------------------------------------------------------------------

    def _descend(self, fresh) -> Optional[SamplingResult]:
        t0 = time.time()
        self.stats.descents += 1
        base = fresh()
        try:
            return self._walk(base)
        finally:
            # Recorded once, here, rather than at each of the seven ways a descent can end.
            self.stats.seconds += time.time() - t0
            self.stats.model_calls += base.forward_passes()

    def _walk(self, base) -> Optional[SamplingResult]:
        state = base
        node = self.trie.root
        depth = 0
        context: List[int] = []
        pending = False
        # raw_logprob holds the pristine envelope ratio and is never rewritten (only log_theta is),
        # so summing the chosen entries telescopes to log E(w$) - log E(root) for free.
        log_ratio_sum = 0.0

        stop_at_bound = self.on_max_length == "stop"
        for _ in range(self.max_new_tokens + (1 if stop_at_bound else 0)):
            at_bound = stop_at_bound and len(context) >= self.max_new_tokens
            if node.raw_logprob is None:
                # First visit: expand. The node's bound E(u) is replaced by the total mass of its
                # children, which can only be smaller, and the descent survives with exactly the
                # ratio between them. This is the only place a rejection can happen.
                ratios = state.forced_stop_ratios() if at_bound else state.log_ratios()

                # In log space throughout. Exponentiating first underflows to exactly zero below
                # about -104 in float32, and a zero total is indistinguishable from an impossible
                # prefix, so a live subtree would be deleted from the support without a word.
                log_survival = float(torch.logsumexp(ratios.double(), dim=0))
                # Checked before anything is written. Raising after the node holds a bound but
                # before that bound reaches its parent would leave a trie that is quietly wrong,
                # so a caller who caught this and carried on would get biased samples forever.
                if log_survival > _LOG_SLACK:
                    raise ValueError(
                        f"envelope condition (ii) violated at depth {depth}: the children of this "
                        f"prefix carry {math.exp(log_survival):.9f} of its bound, which exceeds "
                        "one. The expression is not an envelope for the target and sampling from "
                        "it would not be exact."
                    )

                # The gap between envelope and target at this prefix's complete sequence is a
                # function of the prefix, so it is known here, at the expansion, while the models
                # are materialized -- before the leaf is ever drawn. Computing it here is what
                # keeps a revisited prefix free for every sampler: the gap is a forward pass, and
                # caching it is caching that pass, which the non-adaptive modes get as much as
                # the adaptive one does.
                #
                # What to do with it is where they part. MARS folds it into the bound on the stop
                # token, which restores condition (i) at this leaf, removes the need for any
                # acceptance step there, and -- the point -- charges the slack to the survival
                # coin below, from where `_propagate` carries it to the ancestors like any other
                # tightening. Held back at the leaf instead, this slack is learned from never. In
                # the coverage regime (a mean at tau > 1) that is the whole of it: MARS runs there
                # on the dominating mixture, which meets condition (ii) with equality at every
                # interior prefix, so every bit of envelope slack sits at the leaves. Leaving it
                # there makes the root mass constant and MARS no better than RS on the mixture.
                #
                # The non-adaptive modes must not fold it: a folded gap is a bound that stays
                # tightened for every later descent, and not learning is the whole of what they
                # are. They keep the coin at the leaf, re-flipped per descent off the cached gap.
                # Per descent the two are the same draw -- the fold moves a rejection from the
                # leaf up to the prefix it belongs to, and leaves the probability of reaching any
                # other child untouched -- so what separates the samplers is propagation alone.
                gap = 0.0
                if self.envelope.needs_leaf_rejection:
                    # `at_bound`: stopped here, so the sequence's weight is this prefix's weight,
                    # not the weight of this prefix followed by the marker.
                    gap = (state.log_truncation_acceptance() if at_bound
                           else state.log_leaf_acceptance(state.eos_token_id))
                node.leaf_gap = gap
                if gap < 0.0 and self.adaptive:
                    # `log_ratios` builds a fresh tensor per call, but copy anyway: writing
                    # through to a vector the runtime might one day cache would corrupt the
                    # bounds of every later prefix.
                    ratios = ratios.clone()
                    ratios[state.eos_token_id] += gap
                    log_survival = float(torch.logsumexp(ratios.double(), dim=0))
                    node.leaf_gap = 0.0   # it is in the bound now; nothing left to pay at the leaf
                    self.stats.leaf_tightenings += 1

                node.raw_logprob = ratios.unsqueeze(0).cpu()
                # Kept so that cached rejection sampling can flip this coin again on a revisit
                # without recomputing it. MARS never reads it: after propagation the coin is gone.
                node.log_survival = log_survival
                # The correction vector is zero at every node until a propagation writes to it,
                # and most nodes are never written to. Allocating it eagerly doubled the trie's
                # memory, a megabyte per node at a 128k vocabulary, which is what exhausted a 48 GB
                # job on the long, loose grammars. It is created on first write in `_propagate`.
                node.log_theta = None
                self.stats.expansions += 1
                pending = True
                if math.isnan(log_survival) or log_survival == NEG_INF:
                    self._propagate(node, depth, context)
                    self.stats.dead_ends += 1
                    return None
                if log_survival < 0.0 and math.log(max(torch.rand(()).item(), 1e-300)) > log_survival:
                    self._propagate(node, depth, context)
                    self.stats.rejections += 1
                    self.stats.rejected_mass += max(0.0, 1.0 - math.exp(log_survival))
                    return None
            elif self.cache:
                # A cached prefix under rejection sampling: nothing was learned here, so the
                # descent faces the same survival coin it faced on the first visit.
                log_survival = node.log_survival
                if math.isnan(log_survival) or log_survival == NEG_INF:
                    self.stats.dead_ends += 1
                    return None
                if log_survival < 0.0 and math.log(max(torch.rand(()).item(), 1e-300)) > log_survival:
                    self.stats.rejections += 1
                    self.stats.rejected_mass += max(0.0, 1.0 - math.exp(log_survival))
                    return None

            if self.cache:
                # Under cached rejection sampling a node's bounds never change, so the cumulative
                # table is built on the first visit and every revisit is one uniform and a binary
                # search, instead of a float64 pass over the vocabulary. Same draw, uniform for
                # uniform; without this a timed-out instance spends minutes of CPU on revisits.
                prepared = getattr(node, "prepared", None)
                if prepared is None:
                    prepared = node.prepared = prepare_log(node.raw_logprob[0]) or False
                if prepared is False:
                    self.stats.dead_ends += 1
                    return None
                token = draw_prepared(prepared)
            else:
                bounds = (node.raw_logprob[0] if node.log_theta is None
                          else node.raw_logprob[0] + node.log_theta[0])
                finite = torch.isfinite(bounds)
                if not finite.any():
                    self._propagate(node, depth, context)
                    self.stats.dead_ends += 1
                    return None

                # Not torch.multinomial: its CPU path can return an entry of negligible
                # probability about once per 130 draws at this vocabulary size (see
                # casa.utils.draw).
                token = draw_log(bounds)

            if token == state.eos_token_id:
                # The envelope is exact on complete sequences, so there is nothing left to reject.
                # Under MARS that holds for a dominating envelope too: the expansion folded the
                # gap into the bound stored here, `leaf_gap` is zero, and the rejection it stands
                # for was already paid, as a lower survival probability, at the prefix. The
                # non-adaptive modes did not fold, and pay it here, from the cached gap and with
                # no forward pass.
                # Read the attribute, do not default it: every expansion sets it, and a node
                # that was never expanded cannot be drawn from. Defaulting to zero here would
                # skip the non-adaptive modes' acceptance step in silence: a bias, not a crash.
                log_acc = node.leaf_gap
                if log_acc < 0.0:
                    self.stats.leaf_trials += 1
                    if math.log(max(torch.rand(()).item(), 1e-300)) > log_acc:
                        # Propagate before giving up. An expansion made during this descent has
                        # tightened a bound, and leaving that out of the ancestors would let the
                        # parent keep over-weighting this subtree with no rejection to pay for it,
                        # which biases every later descent.
                        if pending:
                            self._propagate(node, depth, context)
                        self.stats.leaf_rejections += 1
                        return None
                log_ratio_sum += float(node.raw_logprob[0, token])
                if pending:
                    self._propagate(node, depth, context)
                return self._result(base, context, log_ratio_sum, log_acc)

            log_ratio_sum += float(node.raw_logprob[0, token])
            context.append(token)
            state = state.advance(token)
            if token not in node.children:
                node.create_child(token)
            node = node.children[token]
            depth += 1

        # Ran out of length without terminating. Not a sample.
        if pending:
            self._propagate(node, depth, context)
        self.stats.length_cutoffs += 1
        return None

    def _propagate(self, node, depth: int, context: List[int]) -> None:
        """Replace each ancestor's bound on this path by the total mass of its children.

        Identical to the CARS update, and correct for the same reason: the overwrite is valid
        because ``raw_logprob`` already holds the full envelope ratio, so the parent's entry for
        this child is its bound and nothing else.
        """
        if self.cache:
            # Rejection sampling learns nothing; the cache keeps ratios, never tightened bounds.
            return
        # A node created but never expanded holds no bound of its own; its parent's entry for it
        # is still the original envelope value, so there is nothing to push up from it. This
        # happens whenever a descent stops on the length bound.
        while node.raw_logprob is None and depth > 0:
            depth -= 1
            node = node.parent
        while depth > 0:
            bounds = (node.raw_logprob[0] if node.log_theta is None
                      else node.raw_logprob[0] + node.log_theta[0])
            total = torch.logsumexp(bounds, dim=0)
            depth -= 1
            node = node.parent
            if node.log_theta is None:
                node.log_theta = torch.zeros(1, node.raw_logprob.shape[-1])
            node.log_theta[0, context[depth]] = total

    def _result(self, base: "rt.EnvelopeRT", context: List[int],
                log_ratio_sum: float, log_leaf_gap: float = 0.0) -> SamplingResult:
        """Package a yielded sequence.

        ``raw_logprob`` is ``log phi(w$)``, the target's own unnormalized weight. The ratios
        stored in the trie telescope to ``log E(w$)``, which equals it by condition (i). In the
        coverage regime the envelope starts out dominating, for which that would not hold; MARS
        tightens the bound on each stop token to the target's own weight as it expands, so the sum
        needs no correction, while the non-adaptive modes leave the difference at the leaf and it
        is added back here. Getting that wrong makes every reweighting or divergence computed from these
        numbers wrong too.

        ``constrained_logprob`` is the same quantity relative to the envelope at the root. Note
        that CARS puts the descent's own log-probability in this field, which is a different thing:
        it differs from this by the current root mass.

        Neither costs a model call; both come from values already in hand.
        """
        tok = base.models[0].llm.tokenizer
        ids = context + [base.eos_token_id]
        root_w = base.log_weight()
        log_phi = root_w + log_ratio_sum + log_leaf_gap
        return SamplingResult(
            tokens=[tok.decode([t]) for t in ids],
            token_ids=ids,
            text=tok.decode(context),
            raw_logprob=float(log_phi),
            constrained_logprob=float(log_phi - root_w),
            success=True,
        )


class _Stats:
    """What a run cost, in the quantities the theory talks about."""

    __slots__ = ("descents", "accepted", "expansions", "rejections", "rejected_mass",
                 "leaf_tightenings", "leaf_trials", "leaf_rejections",
                 "dead_ends", "length_cutoffs",
                 "timeouts", "seconds", "model_calls")

    def __init__(self):
        self.descents = 0
        self.accepted = 0
        self.expansions = 0
        self.rejections = 0
        self.rejected_mass = 0.0
        # Expansions at which a dominating envelope's leaf gap was folded into the bound
        # (adaptive), against leaves where it was paid as a coin flip instead (non-adaptive).
        self.leaf_tightenings = 0
        self.leaf_trials = 0
        self.leaf_rejections = 0
        self.dead_ends = 0
        self.length_cutoffs = 0
        self.timeouts = 0
        self.seconds = 0.0
        self.model_calls = 0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self) -> str:
        return f"_Stats({self.as_dict()})"
