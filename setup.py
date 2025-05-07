from setuptools import setup, find_packages

setup(
    name="branch_reasoning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "wandb>=0.15.0",
        "pandas>=2.0.0",
        "numpy>=1.23.0",
        "matplotlib>=3.7.0",
    ],
    description="Tools for branch reasoning in language models",
    author="Branch Reasoning Team",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)