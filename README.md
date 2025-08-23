# Gemma 3 270M Fine-tuning with MLX

A clean, minimal setup for fine-tuning **Gemma 3 270M** with **LoRA** on Apple Silicon using **MLX**. Supports both **Alpaca** (general instruction following) and **SQL** (text-to-SQL generation) datasets, but you can use any dataset you want.

## ✨ Features

- ⚡ **Apple Silicon Optimized**: Uses MLX framework for M-series chips
- 🎯 **4 Core Functions**: Dataset → Train → Test → Benchmark
- 🔧 **Easy Setup**: Single pipenv environment with pinned dependencies
- 🔄 **Two Sample Dataset**: Alpaca (52K examples) or Text-to-SQL (106K examples)  

## 🚀 Quick Setup

### Prerequisites
```bash
# macOS with Apple Silicon
xcode-select --install
brew install python@3.11 git
```

### 1. Environment Setup
```bash
# Clone and setup
git clone https://github.com/acrosa/mlx-gemma3-270m
cd gemma3-270-finetune

# Create virtual environment  
pipenv --python 3.11

# Install dependencies
pipenv run pip install mlx mlx-lm datasets pandas transformers huggingface-hub
```

### 2. Environment Variables
Create `.env` file:
```bash
# Required: Get token from https://huggingface.co/settings/tokens
HUGGINGFACE_HUB_TOKEN=hf_your_token_here

# Optional: Faster downloads
HF_HUB_ENABLE_HF_TRANSFER=1
TOKENIZERS_PARALLELISM=false
```

### 3. Model Access
1. Visit https://huggingface.co/google/gemma-3-270m-it
2. **Request Access** (required for gated model)
3. Verify access: `pipenv run python -c "from huggingface_hub import whoami; print(whoami())"`

## 📋 The 4 Core Functions

### 1️⃣ **Prepare Dataset**
```bash
# SQL Dataset (106K examples, text-to-SQL)
pipenv run python dataset.py --type sql --sample-size 1000

# Custom SQL dataset
pipenv run python dataset.py --type sql --dataset-name your-username/your-sql-dataset

# Alpaca Dataset (52K examples, general instruction following)  
pipenv run python dataset.py --type alpaca --sample-size 1000

# Custom Alpaca-style dataset
pipenv run python dataset.py --type alpaca --dataset-name microsoft/orca-math-word-problems-200k

# Full datasets (no sample limit)
pipenv run python dataset.py --type sql
pipenv run python dataset.py --type alpaca
```

**Output**: Creates `data/train.jsonl` and `data/valid.jsonl`

💡 **Custom Dataset Requirements:**
- **SQL datasets**: Must have columns: `sql_prompt`, `sql`, `sql_context` (optional), `sql_explanation` (optional)  
- **Alpaca datasets**: Must have columns: `instruction`, `output`, `input` (optional)

### 2️⃣ **Train Model**
```bash
# Quick test training (fast, for testing)
pipenv run python train.py --dataset-type sql --quick-test
pipenv run python train.py --dataset-type alpaca --quick-test

# Standard training  
pipenv run python train.py --dataset-type sql
pipenv run python train.py --dataset-type alpaca
```

**Output**: Creates model adapters in `checkpoints/` folder

### 3️⃣ **Test Models** (Base vs Fine-tuned)
```bash
# Interactive chat - fine-tuned model only
pipenv run python test_finetuned.py --adapter-path checkpoints --interactive

# Interactive chat - BOTH models side-by-side (recommended to get a sense!)
pipenv run python test_finetuned.py --adapter-path checkpoints --interactive --show-base

# Automated comparison with validation data
pipenv run python test_finetuned.py --adapter-path checkpoints --use-validation --num-samples 5
```

**Output**: Interactive chat or side-by-side response comparison with quality metrics

💡 **Pro Tip**: Use `--show-base` to see how fine-tuning improved your model!

**Interactive Chat Output Example:**
```
💬 You: Create a SQL query to find customers in New York
------------------------------------------------------------
🔹 Base Model:
SELECT * FROM database WHERE location...
⚡ 150.2 tok/s, 0.85s

------------------------------
🔸 Fine-tuned Model:
SELECT * FROM customers WHERE city = 'New York';

Explanation: This query selects all customer records...
⚡ 145.8 tok/s, 0.92s
------------------------------------------------------------
```

### 4️⃣ **Benchmark Performance**
```bash  
# Performance benchmarking (speed, memory, throughput)
pipenv run python benchmark.py --adapter-path checkpoints --num-samples 10

# Compare base vs fine-tuned performance metrics
pipenv run python benchmark.py --adapter-path checkpoints --compare --num-samples 10

# Quick performance benchmark
pipenv run python benchmark.py --adapter-path checkpoints --quick --num-samples 5
```

**Output**: Speed (tokens/sec), memory usage, and throughput analysis with numerical metrics

## 🔄 Testing vs Benchmarking

**`test_finetuned.py`** - **Qualitative Analysis** 📝
- **Interactive chat mode**: `--interactive` for conversational testing
- **Automated comparison**: Side-by-side base vs fine-tuned responses  
- Quality metrics (coherence, format, content)
- Great for seeing **what** the model learned
- Use validation data: `--use-validation --num-samples 5`

**`benchmark.py`** - **Performance Analysis** ⚡  
- Speed, memory, throughput measurements
- Technical performance metrics
- Great for seeing **how fast** the model runs
- Always uses validation data by default

## 📊 Dataset Information

### **SQL Dataset** (`gretelai/synthetic_text_to_sql`)
- **Size**: 106,000 examples
- **Domains**: 100+ (retail, healthcare, finance, etc.)  
- **Complexity**: Basic, intermediate, advanced, analytical queries
- **Format**: Natural language → SQL + explanation
- **Use case**: Text-to-SQL generation, database querying

### **Alpaca Dataset** (`yahma/alpaca-cleaned`)
- **Size**: 52,002 examples  
- **Content**: Instruction-following tasks
- **Quality**: Cleaned responses, diverse topics
- **Format**: Instruction + optional input → response  
- **Use case**: General-purpose assistant, instruction following

### **Alternative Datasets You Can Use**

**SQL Datasets:**
- `gretelai/synthetic_text_to_sql` - Default, 106K examples, multi-domain
- `Clinton/Text-to-sql-v1` - 78K examples with explanations
- `knowrohit07/know_sql` - 17K examples with context

**Alpaca-Style Datasets:**  
- `yahma/alpaca-cleaned` - Default, 52K instruction-following examples
- `microsoft/orca-math-word-problems-200k` - Math-focused instructions
- `garage-bAInd/Open-Platypus` - STEM-focused instruction dataset
- `teknium/OpenHermes-2.5` - High-quality instruction following

**Usage Example:**
```bash
# Use alternative SQL dataset
pipenv run python dataset.py --type sql --dataset-name Clinton/Text-to-sql-v1 --sample-size 5000

# Use math-focused Alpaca dataset  
pipenv run python dataset.py --type alpaca --dataset-name microsoft/orca-math-word-problems-200k
```

## ⚙️ Configuration Options

### **Training Modes**
| Mode | Iterations | Batch Size | Learning Rate | Use Case |
|------|------------|------------|---------------|----------|
| `--quick-test` | 50-100 | 2 | 2e-4 | Testing setup |
| Standard | 1500-2000 | 4 | 2e-4 | Regular training |
| `--conservative` | 600-800 | 4 | 5e-5 | Anti-overfitting |

### **Custom Parameters**
```bash
# Custom training parameters
pipenv run python train.py \
  --dataset-type sql \
  --sample-size 5000 \
  --max-iters 1000 \
  --batch-size 4 \
  --learning-rate 1e-4

# Train with custom dataset (prepare data first!)
pipenv run python dataset.py --type sql --dataset-name Clinton/Text-to-sql-v1
pipenv run python train.py --dataset-type sql
```

## 🎯 Example Workflows

### **Quick Test Everything**
```bash
# 1. Prepare small SQL dataset
# this will download and use the `text-to-sql` dataset `gretelai/synthetic_text_to_sql`
pipenv run python dataset.py --type sql --sample-size 100

# 2. Quick training
pipenv run python train.py --dataset-type sql --quick-test  

# 3. Test interactively (see both base and fine-tuned responses!)
pipenv run python test_finetuned.py --adapter-path checkpoints --interactive --show-base

# 4. Benchmark performance (speed/memory metrics)
pipenv run python benchmark.py --adapter-path checkpoints --num-samples 5
```

### **Full training**
```bash
# 1. Full SQL dataset
pipenv run python dataset.py --type sql

# 2. Standard training (2000 iterations)
pipenv run python train.py --dataset-type sql

# 3. Interactive testing (chat with your model!)
pipenv run python test_finetuned.py --adapter-path checkpoints --interactive --show-base

# 4. Performance analysis (speed/memory comparison)
pipenv run python benchmark.py --adapter-path checkpoints --compare --num-samples 10
```

### **Side by side comparison**

Below you can see sample output for the interactive script testing a fine tune on the SQL dataset.

```bash
pipenv run python test_finetuned.py --adapter-path checkpoints --interactive --show-base
```

Prompt the script and you will get both responses, the base and fine tuned model:
```
Loading .env environment variables...
Evaluation Configuration:
========================================
Model: google/gemma-3-270m-it
Adapter Path: checkpoints
Max Tokens: 256
Temperature: 0.7
Number of Test Prompts: 8
========================================
Loading base model...
Loading fine-tuned model with adapters: checkpoints

================================================================================
🤖 INTERACTIVE CHAT MODE
================================================================================
Chat with your fine-tuned model! Type your questions below.
Commands: 'quit', 'exit', 'bye' to stop
💡 Showing both base and fine-tuned responses for comparison
================================================================================

💬 You: What is the number of recyclable and non-recyclable materials for each origin, and the percentage of recyclable materials for each origin?\nContext: CREATE TABLE mateials (id INT PRIMARY KEY, name VARCHAR(255), origin VARCHAR(255), recyclable BOOLEAN); INSERT INTO materials (id, name, origin, recyclable) VALUES (1, 'Plastic', 'China', FALSE), (2, 'Aluminum', 'Canada', TRUE), (3, 'Glass', 'Mexico', TRUE), (4, 'Paper', 'India', TRUE);

------------------------------------------------------------
🔹 Base Model:
Here's the breakdown of the data:

*   **Recyclable Materials:**
    *   China: 100%
    *   Mexico: 100%
    *   India: 100%

*   **Non-Recyclable Materials:**
    *   Plastic: 50%
    *   Aluminum: 20%
    *   Glass: 10%<end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn>
<end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn>
<end_of_turn><end_of_turn><end_of_turn><end_of_turn>
<end_of_turn><end_of_turn><end_of_turn><end_of_turn>
<end_of_turn><end_of_turn><end_of_turn>
<end_of_turn><end_of_turn><end_of_turn>
<end_of_turn><end_of_turn><end_of_turn>
<end_of_turn><end_of_turn>
<end_of_turn><end_of_turn>
<end_of_turn><end_of_turn><end_of_turn>
<end_of_turn><end_of_turn><end_of_turn>
<end_of_turn><end_of_turn><end_of_turn>
<end_of_turn><end_of_turn><end_of_turn>
<end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn>
<end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn>
<end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn>
<end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn><end_of_turn>
⚡ 152.3 tok/s, 1.69s

------------------------------
🔸 Fine-tuned Model:
SELECT origin, COUNT(*) as num_recyclable, (CASE WHEN recyclable THEN 1 ELSE 0 END) as percentage FROM materials WHERE origin IN (SELECT origin FROM materials)) as recycling_percentage FROM materials WHERE origin IN (SELECT origin FROM materials)

Explanation: The SQL query calculates the number of recyclable and non-recyclable materials for each origin, and the percentage of recyclable materials for each origin. It does this by selecting the origin and counting the number of recyclable and non-recyclable materials for each origin. The result is then displayed in the material table.<end_of_turn>
⚡ 151.1 tok/s, 0.77s
------------------------------------------------------------
```