from casa import LLM, Grammar, MARS
from casa.algebra import Model, constrain, intersect
from casa.utils.envelope_runtime import grammar_mask

llm = LLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

grammar_str = """
start: CHARACTER " " ACTION " " LOCATION "."
CHARACTER: "a dragon" | "a knight" | "a wizard"
ACTION: "discovered" | "protected" | "enchanted"
LOCATION: "the castle" | "the forest" | "the treasure"
"""

prompt = "Once upon a time,"
grammar = Grammar.from_string(grammar_str, llm.tokenizer)

# A grammar constraint is one expression of the algebra; this is exactly the CARS target.
target = constrain(Model(llm), grammar_mask(grammar))
sampler = MARS(target, max_new_tokens=32, verbose=True)
results = sampler.sample(prompt, n_samples=10, max_attempts=100)

if results:
	print("\nGenerated samples,")
	for i, result in enumerate(results, 1):
		print(f"  {i}. {prompt} {result.text}")
else:
	print("Failed to generate any samples")

# Two prompts, one model: completions that are probable under both, and valid under the grammar.
both = intersect(Model(llm, prompt=prompt), Model(llm, prompt="In a land of dragons,"))
sampler = MARS(constrain(both, grammar_mask(grammar)), max_new_tokens=32, verbose=True)
results = sampler.sample(n_samples=10, max_attempts=100)

if results:
	print("\nGenerated samples (two prompts),")
	for i, result in enumerate(results, 1):
		print(f"  {i}. {result.text}")
else:
	print("Failed to generate any samples")
