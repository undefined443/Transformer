# Transformer

A TensorFlow implementation of the Transformer architecture, adapted from the Coursera assignment [Transformers Architecture with TensorFlow](https://www.coursera.org/learn/nlp-sequence-models/programming/roP5y/transformers-architecture-with-tensorflow).

## Project Structure

- `src/encoder.py` - Encoder and EncoderLayer implementations
- `src/decoder.py` - Decoder and DecoderLayer implementations
- `src/transformer.py` - Main Transformer model
- `src/utils.py` - Utility functions (positional encoding, attention, etc.)

## Requirements

- Python >= 3.13
- TensorFlow >= 2.21.0
- NumPy >= 2.4.6

## Installation

Use [uv](https://github.com/astral-sh/uv) to manage dependencies:

```sh
uv sync
```

## Testing

Run all tests:

```sh
uv run -m pytest tests/
```

Run a specific test:

```sh
uv run -m pytest tests/test_all.py::test_get_angles -v
```

## Notes

- Numerical precision may vary across different GPU hardware and TensorFlow versions, which could cause tests to fail.
- It's recommended to use `uv sync` to create a consistent development environment for stable test results.
