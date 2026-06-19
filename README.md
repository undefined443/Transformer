# Transformer

Transformer 架构的 TensorFlow 实现，改编自 Coursera 课程作业 [Transformers Architecture with TensorFlow](https://www.coursera.org/learn/nlp-sequence-models/programming/roP5y/transformers-architecture-with-tensorflow)。

## 项目结构

- `src/encoder.py` - Encoder 和 EncoderLayer 实现
- `src/decoder.py` - Decoder 和 DecoderLayer 实现
- `src/transformer.py` - Transformer 模型主体
- `src/utils.py` - 辅助函数（位置编码、attention 等）

## 环境要求

- Python >= 3.13
- TensorFlow >= 2.21.0
- NumPy >= 2.4.6

## 安装依赖

使用 [uv](https://github.com/astral-sh/uv) 管理依赖：

```sh
uv sync
```

## 测试

运行所有测试：

```sh
uv run -m pytest tests/
```

运行特定测试：

```sh
uv run -m pytest tests/test_all.py::test_get_angles -v
```

## 注意事项

- 由于不同的 GPU 硬件和 TensorFlow 版本，计算结果可能存在数值精度差异，导致测试不通过。
- 建议使用 `uv sync` 创建一致的开发环境，以获得稳定的测试结果。
