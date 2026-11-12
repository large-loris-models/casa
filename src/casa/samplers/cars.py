import time
from typing import List, Optional

import torch

from casa.samplers.base import SamplingResult
from casa.utils.draw import draw_probs
from casa.utils.oracle_trie import Trie
from casa.utils.helpers import print_progress


class CARS:
    """Constrained Adaptive Rejection Sampling (CARS).

    Combines adaptive learning with first token constraints for optimal efficiency.
    Uses KV-cached token-by-token decoding with a cross-attempt oracle trie by
    default; pass ``fast=False`` for the ``model.generate``-based implementation.
    """

    def __init__(self, llm, grammar, max_new_tokens: int = 512,
                 verbose: bool = False, temperature: float = 1.0, fast: bool = True,
                 asap: bool = False):
        self.llm = llm
        self.grammar = grammar
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.temperature = temperature
        self.fast = fast
        self.asap = asap
        self.device = llm.device
        self.eos_token_id = llm.tokenizer.eos_token_id
        self.vocab_size = grammar.recognizer.vocab_size
        self.trie = Trie()
        self.logits_process_time = 0.0

    def _encode_prompt(self, prompt: str) -> List[int]:
        formatted = self.llm.format_prompt(prompt)
        return self.llm.tokenizer.encode(formatted, add_special_tokens=False)

    def _forward(self, token_ids: List[int], past, cached_len: int):
        if past is not None and cached_len > 0:
            new_ids = torch.tensor([token_ids[cached_len:]], device=self.device)
        else:
            new_ids = torch.tensor([token_ids], device=self.device)
            past = None
        with torch.no_grad():
            out = self.llm.model(new_ids, past_key_values=past, use_cache=True)
        logits = out.logits[0, -1, :].float()
        return logits, out.past_key_values, len(token_ids)

    def sample(self, prompt: str, n_samples: int = 1,
               max_attempts: int = 100) -> List[SamplingResult]:
        if not self.fast:
            from casa.samplers.rejection import _GenerateCARS
            return _GenerateCARS(
                self.llm, self.grammar, self.max_new_tokens,
                verbose=self.verbose, temperature=self.temperature, asap=self.asap,
            ).sample(prompt, n_samples, max_attempts)

        prompt_ids = self._encode_prompt(prompt)
        results: List[SamplingResult] = []
        self.trie = Trie()

        for sample_idx in range(n_samples):
            n_attempts = 0
            success = False
            for _ in range(max_attempts):
                n_attempts += 1
                result = self._generate_one(prompt_ids)
                if result is not None:
                    result.attempts = n_attempts
                    results.append(result)
                    print_progress(sample_idx + 1, n_samples, n_attempts,
                                   max_attempts, self.verbose, timeout=False)
                    success = True
                    break
            if not success:
                print_progress(sample_idx + 1, n_samples, n_attempts,
                               max_attempts, self.verbose, timeout=True)

        return results

    def _generate_one(self, prompt_ids: List[int]) -> Optional[SamplingResult]:
        start_time = time.time()
        rec = self.grammar.recognizer
        rec.reset()

        past, cached_len = None, 0
        context: List[int] = []
        node = self.trie.root
        depth = 0
        raw_lps: List[float] = []
        cons_lps: List[float] = []
        recompute_needed = False

        for step in range(self.max_new_tokens):
            is_root = step == 0

            if not is_root:
                if not rec.try_advance_token_ids(torch.tensor(context)):
                    self._reject(node, depth, context, context[-1])
                    self.logits_process_time += time.time() - start_time
                    return None

                last = context[-1]
                if last not in node.children:
                    node.create_child(last)
                node = node.children[last]
                depth += 1

            if node.raw_logprob is not None:
                raw = node.raw_logprob[0].to(self.device)
                is_new_node = False
            else:
                logits, past, cached_len = self._forward(
                    prompt_ids + context, past, cached_len)
                if logits.shape[-1] > self.vocab_size:
                    logits[self.vocab_size:] = float("-inf")
                if self.temperature != 1.0:
                    logits = logits / self.temperature
                raw = torch.log_softmax(logits, dim=-1)
                node.raw_logprob = raw.unsqueeze(0).cpu()
                node.log_theta = torch.zeros(1, raw.shape[-1])
                rec.apply_token_bitmask(node.log_theta, rec.filter_vocab())
                is_new_node = True
                recompute_needed = True

            adjust = self.asap or is_root or not is_new_node
            if adjust:
                sampling_logits = raw + node.log_theta[0].to(self.device)
            else:
                sampling_logits = raw

            probs = torch.softmax(sampling_logits, dim=-1)
            if not torch.isfinite(probs).all() or probs.sum() < 1e-10:
                self.logits_process_time += time.time() - start_time
                return None
            # Not torch.multinomial: its CPU path can return a token of negligible probability
            # about once per 130 draws at this vocabulary size (see casa.utils.draw).
            next_token = draw_probs(probs)

            raw_lps.append(raw[next_token].item())
            cons_lps.append(torch.log_softmax(sampling_logits, dim=-1)[next_token].item())

            if next_token == self.eos_token_id:
                if rec.try_advance_token_ids(torch.tensor(context + [next_token])):
                    if recompute_needed:
                        self._recompute(node, depth, context)
                    self.logits_process_time += time.time() - start_time
                    return self._make_result(context, raw_lps, cons_lps)
                self._reject(node, depth, context, next_token)
                self.logits_process_time += time.time() - start_time
                return None

            context.append(next_token)

        if not rec.try_advance_token_ids(torch.tensor(context)):
            self._reject(node, depth, context, context[-1])
            self.logits_process_time += time.time() - start_time
            return None
        if rec.is_accepting():
            if recompute_needed:
                self._recompute(node, depth, context)
            self.logits_process_time += time.time() - start_time
            return self._make_result(context, raw_lps, cons_lps, eos=False)
        self._reject(node, depth, context, context[-1])
        self.logits_process_time += time.time() - start_time
        return None

    def _make_result(self, context, raw_lps, cons_lps, eos: bool = True) -> SamplingResult:
        token_ids = context + [self.eos_token_id] if eos else list(context)
        tokens = [self.llm.tokenizer.decode([t]) for t in token_ids]
        text = self.llm.tokenizer.decode(context)
        return SamplingResult(
            tokens=tokens,
            token_ids=token_ids,
            text=text,
            raw_logprob=float(sum(raw_lps)),
            constrained_logprob=float(sum(cons_lps)),
            success=True,
        )

    def _reject(self, node, depth, context, failed_token) -> None:
        node.log_theta[0, failed_token] = float("-inf")
        self._recompute(node, depth, context)

    def _recompute(self, node, depth, context) -> None:
        while depth > 0:
            new_log_theta = torch.log(
                torch.exp(node.raw_logprob[0] + node.log_theta[0]).sum()
            )
            depth -= 1
            node = node.parent
            node.log_theta[0, context[depth]] = new_log_theta


class ASAp(CARS):
    """Adaptive Sampling with Approximate expected futures (ASAp).

    Grammar-Aligned Decoding (arXiv:2405.21047). Applies the grammar mask and
    learned EFG correction at every step, so samples are grammatical by
    construction and the empirical distribution converges to the
    grammar-conditioned LM distribution across draws.
    """

    def __init__(self, llm, grammar, max_new_tokens: int = 512,
                 verbose: bool = False, temperature: float = 1.0, fast: bool = True):
        super().__init__(llm, grammar, max_new_tokens, verbose=verbose,
                         temperature=temperature, fast=fast, asap=True)
