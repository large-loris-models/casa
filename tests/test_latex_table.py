from casa import LLM, Grammar, CARS

# 使用开放模型，无需 Hugging Face 授权
llm = LLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 定义 LaTeX 矩阵/表格的语法约束
# 支持 pmatrix, bmatrix, array 等格式
grammar_str = """
start: matrix_start matrix_content matrix_end
matrix_start: "\\\\begin{" matrix_type "}"
matrix_type: "pmatrix" | "bmatrix" | "array"
matrix_content: row (row_sep row)*
row: cell (col_sep cell)*
cell: number | negative_number
number: DIGIT+
negative_number: "-" DIGIT+
DIGIT: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
col_sep: " & "
row_sep: " \\\\ "
matrix_end: "\\\\end{" matrix_type "}"
"""

# 测试用例：将自然语言描述的矩阵转换为 LaTeX
test_cases = [
    "[1,2],[3,4]",
    "[1,2,3],[4,5,6]",
    "[10,20],[30,40]",
    "[[1, 2], [3, 4]]",
]

grammar = Grammar.from_string(grammar_str, llm.tokenizer)
sampler = CARS(llm, grammar, max_new_tokens=64, verbose=True)

print("=" * 60)
print("LaTeX Matrix Generation Test")
print("=" * 60)

for matrix_input in test_cases:
    # 构造 prompt，让模型将自然语言描述的矩阵转换成 LaTeX
    prompt = f"Convert the matrix {matrix_input} to LaTeX format. The LaTeX matrix should be:"
    
    print(f"\n📝 Input: {matrix_input}")
    print(f"🎯 Prompt: {prompt}")
    
    results = sampler.sample(prompt, n_samples=3, max_attempts=50)
    
    if results:
        print("\n✅ Generated LaTeX matrices:")
        for i, result in enumerate(results, 1):
            latex_output = result.text.strip()
            print(f"   {i}. {latex_output}")
            
            # 验证是否以正确的格式开始和结束
            if latex_output.startswith("\\begin{") and latex_output.endswith("}"):
                print(f"      ✓ Valid LaTeX format")
            else:
                print(f"      ⚠️  May have formatting issues")
    else:
        print("❌ Failed to generate valid LaTeX matrix")
    
    print("-" * 60)

