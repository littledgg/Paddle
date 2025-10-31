# PaddlePaddle 归约操作算子查找工具

## 简介

本工具用于查找和列出 PaddlePaddle 框架中所有的归约操作算子（Reduction Operators）。

归约操作是指将张量的一个或多个维度减少为更小维度或标量的操作，例如求和、求平均、求最大值等。

## 使用方法

### 运行脚本

```bash
cd /path/to/Paddle
python3 tools/find_reduction_operators.py
```

### 输出说明

脚本会输出三个主要部分：

1. **paddle.XXX 归约算子**: 从 `python/paddle/__init__.py` 中提取
2. **paddle.nn.functional.XXX 归约算子**: 从 `python/paddle/nn/functional/__init__.py` 中提取
3. **paddle._C_ops.XXX 归约算子**: 从 `paddle/phi/ops/yaml/ops.yaml` 中提取

每个部分都按照类别分组：
- 统计归约（mean, sum, max, min等）
- 统计量（std, var, median等）
- 逻辑/计数归约（all, any, count_nonzero）
- 累积归约（cumsum, cumprod等）
- 归一化（norm, layer_norm, batch_norm等）
- 池化归约（avg_pool, max_pool等）
- Softmax归约（softmax, log_softmax）
- 其他归约

## 查看文档

完整的归约算子列表和说明文档位于：

```
doc/REDUCTION_OPERATORS.md
```

该文档包含：
- 所有归约算子的详细列表
- 每个算子的描述
- 使用示例
- 注意事项

## 问题说明

本工具回答以下问题：

> paddle.XXX的代码路径与paddle.nn.functional.XXX的代码路径与paddle._C_ops.XXX的路径分别在哪呢？
> 我需要找出这三个路径下所有包含归约操作的算子，你能帮我找出来吗？
> 比如paddle.mean, paddle.nn.functional.softmax, paddle._C_ops.matmul

### 答案：

1. **paddle.XXX 的代码路径**:
   - `python/paddle/__init__.py`
   - 这里定义了 PaddlePaddle 主命名空间下的所有公开 API

2. **paddle.nn.functional.XXX 的代码路径**:
   - `python/paddle/nn/functional/__init__.py`
   - 这里定义了神经网络相关的函数式 API

3. **paddle._C_ops.XXX 的代码路径**:
   - `python/paddle/_C_ops.py` (Python接口)
   - `paddle/phi/ops/yaml/ops.yaml` (算子定义)
   - 这些是底层 C++ 算子的 Python 绑定

### 归约算子总结：

- **paddle.XXX**: 40 个归约算子
- **paddle.nn.functional.XXX**: 23 个归约算子
- **paddle._C_ops.XXX**: 37+ 个归约算子

详细列表请查看 `doc/REDUCTION_OPERATORS.md`

## 技术细节

### 归约操作的定义

本工具将以下类型的操作识别为归约操作：

1. **基本统计归约**: mean, sum, prod, max, min, amax, amin
2. **统计量**: std, var, median, quantile
3. **逻辑归约**: all, any
4. **计数**: numel, count_nonzero
5. **索引归约**: argmax, argmin
6. **累积归约**: cumsum, cumprod, cummax, cummin
7. **范数**: norm, l1_norm, l2_norm, frobenius_norm
8. **归一化**: layer_norm, batch_norm, group_norm, instance_norm
9. **池化**: avg_pool, max_pool, adaptive_avg_pool, adaptive_max_pool
10. **Softmax**: softmax, log_softmax

### 排除的操作

虽然名称中可能包含某些关键字，但以下操作不被认为是归约操作：
- 分布式通信操作（all_gather, all_reduce等）
- 优化器相关操作（adamax等）
- 量化相关操作（fake_quantize等）
- 特殊的融合操作（fused_batch_norm_act等）

## 维护

如需更新文档或修改归约操作的定义，请编辑：
- `tools/find_reduction_operators.py` - 主脚本
- `doc/REDUCTION_OPERATORS.md` - 文档

## 许可证

本工具遵循 PaddlePaddle 项目的 Apache 2.0 许可证。
