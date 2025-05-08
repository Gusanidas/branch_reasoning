# Branch Reasoning

A Python package for experimenting with branching in language model reasoning, primarily for mathematical problem-solving.

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/yourusername/branch_reasoning.git
cd branch_reasoning
pip install -e .
```

## Running Tests

You can run the tests using Python's unittest module:

```bash
python -m unittest discover tests
```

Or if you have pytest installed:

```bash
pytest tests/
```

For specific test files:

```bash
python -m unittest tests/test_branching.py
# or
pytest tests/test_branching.py
```

## Usage

### Training

To run the training script with HuggingFace (default):

```bash
python scripts/train_branch.py --model_name "Qwen/Qwen2.5-3B-Instruct" --device mps
```

To run the training script with vLLM (requires CUDA and vLLM to be installed):

```bash
python scripts/train_branch.py --use_vllm --model_name "Qwen/Qwen2.5-3B-Instruct"
```

Available command-line options:

- `--use_vllm`: Use vLLM for text generation instead of HuggingFace
- `--model_name`: Model name or path (default: "Qwen/Qwen2.5-3B-Instruct")
- `--tokenizer_name`: Tokenizer name or path (defaults to model_name)
- `--device`: Device to use (cuda, cpu, mps) (default: "cuda")
- `--iterations`: Number of training iterations (default: 10)
- `--wandb_logging`: Enable wandb logging

### Creating and Pushing a Dataset

To create and push a dataset to Hugging Face Hub:

```bash
python scripts/create_and_push_dataset.py
```

To load and concatenate with an existing dataset:

```bash
python scripts/create_and_push_dataset.py --load-existing
```

## Project Structure

- `branch_reasoning/`: Main package directory
  - `generation/`: Code for generating completions and branching
  - `models/`: Model loading utilities
  - `utils/`: Helper functions, prompts, and dataset tools
- `scripts/`: Contains executable scripts
- `tests/`: Unit tests for the package