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

To run the training script:

```bash
python scripts/train_branch.py
```

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