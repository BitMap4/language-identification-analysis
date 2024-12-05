<!-- - set up venv
- modify "venv/lib/python3.12/site-packages/fastai/imports/core.py", line 9:
    - remove import of Iterable from collections
    - import Iterable from collections.abc

- setup indic_nlp_project/ -->


# Devanagari Language Identifier

A machine learning system for identifying Hindi and Marathi text written in Devanagari script using multiple feature extraction techniques and classification models for a comparative study. This is my course project for the course CL2 (Computational Linguistics 2).

## Features

- Supports Hindi and Marathi language identification
- Multiple feature extraction methods:
  - Character frequency analysis 
  - Word length statistics
  - Character class distribution (vowels, consonants, matras)
  - N-gram analysis
  - Morphological analysis
  - POS tagging features (optional)
  - TF-IDF features

## Setup

Run `setup.sh` to install required packages and download necessary data files.

## Usage

Run all cells in [model_comparison.ipynb](./model_comparison.ipynb)
> [!WARNING]
> Create a directory named `<data_size>` in case it is not created automatically on running the notebook.

## Results

Results of the model comparison are available in [results/`<data_size>`](./results/) where `<data_size>` is the configured size of the training + testing data used for the models.