![CASA](logo.png)

Constrained aligned sampling algorithms for language models via CARS, MARS, ASAp, GCD, MCMC, and rejection sampling variants.

## Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended)

### Install from source

```bash
# Clone the repository
git clone https://github.com/LargeLorisModels/casa.git
cd casa

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Install package in editable mode
pip install -e .
```

### Using uv (faster)

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Quick Start

```python
from casa import LLM, Grammar, CARS

llm = LLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

grammar_str = """
start: CHARACTER " " ACTION " " LOCATION "."
CHARACTER: "a dragon" | "a knight" | "a wizard"
ACTION: "discovered" | "protected" | "enchanted"
LOCATION: "the castle" | "the forest" | "the treasure"
"""

prompt = "Once upon a time,"
grammar = Grammar.from_string(grammar_str, llm.tokenizer)
sampler = CARS(llm, grammar, max_new_tokens=32, verbose=True)
results = sampler.sample(prompt, n_samples=10, max_attempts=100)

if results:
	print("\nGenerated samples,")
	for i, result in enumerate(results, 1):
		print(f"  {i}. {prompt} {result.text}")
else:
	print("Failed to generate any samples")
```

## Example Output

With `verbose=True`, you'll see the rejection samplers performance in real-time. Running above code,

```
Sample 01/10: ████████████████████████████████████████ 78 attempts
Sample 02/10: ████████ 12 attempts
Sample 03/10: █ 2 attempts
Sample 04/10: █ 1 attempts
Sample 05/10: █ 1 attempts
...

Generated samples:
  1. Once upon a time, a dragon enchanted the castle.
  2. Once upon a time, a dragon enchanted the forest.
  3. Once upon a time, a dragon enchanted the forest.
 ...
```

## Grammar Engines

Grammars can be compiled with either of two backends, selected per `Grammar` via the `engine` flag:

```python
grammar = Grammar.from_string(grammar_str, llm.tokenizer, engine="llguidance")  # default
grammar = Grammar.from_string(gbnf_str, llm.tokenizer, engine="xgrammar")
```

- `llguidance` (default): accepts Lark-style or GBNF grammars.
- `xgrammar`: accepts GBNF/EBNF grammars only (`root ::= ...`).

A GBNF grammar runs on both engines, making them directly comparable.

## Available Samplers

- **CARS**: Constrained Adaptive Rejection Sampling. Exact samples from the grammar-conditioned LM distribution; learns from rejections and constrains the first token.
- **MARS**: Modular Adaptive Rejection Sampling. Exact samples from combinations of models (products, generalized means, verifier reweighting, grammar constraints) built with an algebra of envelopes; its rejection rate decreases across draws. Reduces to CARS on a hard constraint.
- **MCMC**: Markov Chain Monte Carlo sampling. Avaliable variants,
  - _Uniform_ - Randomly resamples from any position. Balances exploration with structural preservation.
  - _Priority_ - Resample higher perplexity regions first. Targets uncertain tokens for refinement.
  - _Restart_ - Generates from scratch. Independent proposals via importance sampling.
- **ARS**: Adaptive Rejection Sampling
- **RSFT**: Rejection Sampling with First Token constraints
- **ASAp**: Adaptive Sampling with Approximate expected futures (Grammar-Aligned Decoding). Rejection-free; the distribution of samples converges to the grammar-conditioned LM distribution across draws.
- **GCD**: Vanilla Grammar-Constrained Decoding. Masks invalid tokens at every step. Fastest, but distorts the LM distribution.
- **RS**: Basic Rejection Sampling

All samplers share the same interface:

```python
from casa import CARS, ASAp, GCD

sampler = ASAp(llm, grammar, max_new_tokens=32)
sampler = GCD(llm, grammar, max_new_tokens=32)
results = sampler.sample(prompt, n_samples=10)
```

MARS takes a *target* instead of `(llm, grammar)`: an expression built with `casa.algebra`. A grammar constraint is one such expression, so the CARS target is a special case.

```python
from casa import MARS
from casa.algebra import Model, constrain, intersect
from casa.utils.envelope_runtime import grammar_mask

target = constrain(Model(llm), grammar_mask(grammar))                 # the CARS target
target = intersect(Model(llm, prompt=p1), Model(llm, prompt=p2))      # probable under both prompts
results = MARS(target, max_new_tokens=32).sample(prompt, n_samples=10)
```

## Running Tests

```bash
# Run the example
python tests/test_cars.py

# Or other samplers
python tests/test_mcmc.py
python tests/test_mars.py

# Exactness checks on small tabular models (no model download, no GPU)
pytest tests/test_mars_exact.py tests/test_draw.py
```

## References

CASA implements the following algorithms from:

- **RS, ARS, RSFT, CARS**  
  _Constrained Adaptive Rejection Sampling_  
  Preprint | [arXiv:2510.01902](https://arxiv.org/abs/2510.01902)

- **MARS**  
  _Modular Adaptive Rejection Sampling_  
  Preprint | arXiv: TBD

- **ASAp**  
  _Grammar-Aligned Decoding_  
  NeurIPS 2024 | [arXiv:2405.21047](https://arxiv.org/abs/2405.21047)

- **MCMC - Uniform, Priority, Restart**  
  _Constrained Sampling for Language Models Should Be Easy: An MCMC Perspective_  
  NeurIPS 2025 | [arXiv:2506.05754](https://arxiv.org/abs/2506.05754)

## License

MIT License
