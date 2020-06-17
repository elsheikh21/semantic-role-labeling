# Changelog

All notable changes to this project will be documented in this file.

## [Pushed]

## [0.0.6] - 2020-06-08

### Added

- Multi input models
- Compute Epoch Time
- tqdm bars for evaluation and indexing of data
- Add tensorboard logger to training 

### Changed

- Fixed error with creating vocab dictionaries
- `pad_batch()` functionality
- Hyper-parameters vars
- Refactoring hyper-parameters `_print_info()`
- Training verbose format
- Refactoring utilities and training scripts

### Removed

- None

## [0.0.5] - 2020-06-01

### Added

- Early Stopping
- Save model checkpoint, and Load checkpoint

### Changed

- refactoring `main.py`

### Removed

- None

## [0.0.4] - 2020-05-31

### Added

- Model
- Vocabulary
- Evaluation functionality
- Evaluate based on `BATCH_SIZE` in `main.py`

### Changed

- Increase readability of model number of parameters in model summary
- Print model summary & Hyper-parameters via `VERBOSE` flag in `main.py`

### Removed

- predict sentences model's functionality

## [0.0.3] - 2020-05-26

### Added

- Seed
- Decode predictions to JSONDataParser
- Model functionality predict_sentence(s)
- This Changelog

### Changed

- modifying model saving after training functionality

## [0.0.2] - 2020-05-26

### Added

- git-lfs to track model files
- New functionalities to JSONDataParser
  - `get_sentences(*args)` to fetch `data_x, data_y`
  - `build_vocabulary()`
  - `build_label_vocabulary()`
  - `encode_dataset()`
  - `pad_batch`
- JSONDataParser's Utilities file `save_pickle()`, `load_pickle()`
- `main.py` to act as pipeline
- Hyperparameters Object
- Baseline Model Object
- Trainer Object

### Changed

- Modifications to `.gitignore`
- Renaming DataLoader to JSONDataParser

## [0.0.1] - 2020-05-25

### Added

- Directory Structure
- Main files were added
- DataLoader was built with functionality to read dataset
- `requirements.txt` file was added

### Changed

- Modifications to `.gitignore`
