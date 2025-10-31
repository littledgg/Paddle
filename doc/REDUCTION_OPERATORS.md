# PaddlePaddle 归约操作算子详细列表

**Detailed List of Reduction Operators in PaddlePaddle**

本文档列出了 PaddlePaddle 中三个主要路径下的所有归约操作算子。

## 概述

归约操作（Reduction Operations）是指将张量的一个或多个维度减少为更小维度或标量的操作，例如求和、求平均、求最大值等。

## 三个路径说明

### 1. paddle.XXX
- **路径**: `python/paddle/__init__.py`
- **说明**: 这是 PaddlePaddle 的主命名空间，包含了最常用的 API
- **示例**: `paddle.mean()`, `paddle.sum()`, `paddle.max()`

### 2. paddle.nn.functional.XXX
- **路径**: `python/paddle/nn/functional/__init__.py`
- **说明**: 包含神经网络相关的函数式 API，特别是激活函数、归一化和池化操作
- **示例**: `paddle.nn.functional.softmax()`, `paddle.nn.functional.avg_pool2d()`

### 3. paddle._C_ops.XXX
- **路径**: `python/paddle/_C_ops.py`
- **算子定义**: `paddle/phi/ops/yaml/ops.yaml`
- **说明**: 这是底层 C++ 算子的 Python 接口，动态加载自:
  - `paddle.base.core.eager.ops` (动态图)
  - `paddle.base.core.pir.ops` (PIR)
- **示例**: `paddle._C_ops.mean()`, `paddle._C_ops.sum()`

## 一、paddle.XXX 归约算子

**共 40 个归约算子**

### 统计归约 (Statistical Reduction)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 1 | paddle.amax | 沿指定维度返回最大值 |
| 2 | paddle.amin | 沿指定维度返回最小值 |
| 3 | paddle.argmax | 返回最大值的索引 |
| 4 | paddle.argmin | 返回最小值的索引 |
| 5 | paddle.cartesian_prod | 笛卡尔积 |
| 6 | paddle.einsum | 爱因斯坦求和约定（可执行归约） |
| 7 | paddle.fmax | 逐元素比较取最大值（NaN处理） |
| 8 | paddle.fmin | 逐元素比较取最小值（NaN处理） |
| 9 | paddle.logsumexp | log空间的和（log-sum-exp技巧） |
| 10 | paddle.max | 返回张量的最大值 |
| 11 | paddle.maximum | 逐元素比较取最大值 |
| 12 | paddle.mean | 计算张量元素的平均值 |
| 13 | paddle.min | 返回张量的最小值 |
| 14 | paddle.minimum | 逐元素比较取最小值 |
| 15 | paddle.nanmean | 计算平均值（忽略NaN） |
| 16 | paddle.nansum | 计算和（忽略NaN） |
| 17 | paddle.prod | 计算张量元素的乘积 |
| 18 | paddle.sum | 计算张量元素的和 |

### 统计量 (Statistics)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 19 | paddle.median | 计算中位数 |
| 20 | paddle.nanmedian | 计算中位数（忽略NaN） |
| 21 | paddle.nanquantile | 计算分位数（忽略NaN） |
| 22 | paddle.quantile | 计算分位数 |
| 23 | paddle.std | 计算标准差 |
| 24 | paddle.var | 计算方差 |

### 逻辑/计数归约 (Logical/Counting Reduction)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 25 | paddle.all | 检查所有元素是否为真 |
| 26 | paddle.any | 检查是否存在元素为真 |
| 27 | paddle.count_nonzero | 计算非零元素数量 |

### 累积归约 (Cumulative Reduction)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 28 | paddle.cummax | 累积最大值 |
| 29 | paddle.cummin | 累积最小值 |
| 30 | paddle.cumprod | 累积乘积 |
| 31 | paddle.cumprod_ | 累积乘积（原地操作） |
| 32 | paddle.cumsum | 累积和 |
| 33 | paddle.cumsum_ | 累积和（原地操作） |
| 34 | paddle.logcumsumexp | log空间的累积和 |

### 归一化 (Normalization)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 35 | paddle.norm | 计算范数 |
| 36 | paddle.renorm | 重新归一化 |
| 37 | paddle.renorm_ | 重新归一化（原地操作） |

### 其他归约 (Other Reductions)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 38 | paddle.dist | 计算两个张量的距离 |
| 39 | paddle.numel | 返回张量元素总数 |
| 40 | paddle.trace | 计算矩阵的迹（对角线元素和） |

## 二、paddle.nn.functional.XXX 归约算子

**共 23 个归约算子**

### 统计归约 (Statistical Reduction)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 1 | paddle.nn.functional.maxout | Maxout激活（通道维度取最大） |

### 归一化 (Normalization)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 2 | paddle.nn.functional.batch_norm | 批归一化 |
| 3 | paddle.nn.functional.group_norm | 组归一化 |
| 4 | paddle.nn.functional.instance_norm | 实例归一化 |
| 5 | paddle.nn.functional.layer_norm | 层归一化 |
| 6 | paddle.nn.functional.local_response_norm | 局部响应归一化 |

### 池化归约 (Pooling Reduction)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 7 | paddle.nn.functional.adaptive_avg_pool1d | 1维自适应平均池化 |
| 8 | paddle.nn.functional.adaptive_avg_pool2d | 2维自适应平均池化 |
| 9 | paddle.nn.functional.adaptive_avg_pool3d | 3维自适应平均池化 |
| 10 | paddle.nn.functional.adaptive_max_pool1d | 1维自适应最大池化 |
| 11 | paddle.nn.functional.adaptive_max_pool2d | 2维自适应最大池化 |
| 12 | paddle.nn.functional.adaptive_max_pool3d | 3维自适应最大池化 |
| 13 | paddle.nn.functional.avg_pool1d | 1维平均池化 |
| 14 | paddle.nn.functional.avg_pool2d | 2维平均池化 |
| 15 | paddle.nn.functional.avg_pool3d | 3维平均池化 |
| 16 | paddle.nn.functional.lp_pool1d | 1维Lp池化 |
| 17 | paddle.nn.functional.lp_pool2d | 2维Lp池化 |
| 18 | paddle.nn.functional.max_pool1d | 1维最大池化 |
| 19 | paddle.nn.functional.max_pool2d | 2维最大池化 |
| 20 | paddle.nn.functional.max_pool3d | 3维最大池化 |

### Softmax归约 (Softmax Reduction)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 21 | paddle.nn.functional.log_softmax | Log Softmax激活函数 |
| 22 | paddle.nn.functional.softmax | Softmax激活函数 |
| 23 | paddle.nn.functional.softmax_ | Softmax激活函数（原地操作） |

## 三、paddle._C_ops.XXX 归约算子

**共 37 个归约算子**（从 ops.yaml 定义文件中提取）

### 统计归约 (Statistical Reduction)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 1 | paddle._C_ops.amax | 沿指定维度返回最大值 |
| 2 | paddle._C_ops.amin | 沿指定维度返回最小值 |
| 3 | paddle._C_ops.argmax | 返回最大值的索引 |
| 4 | paddle._C_ops.argmin | 返回最小值的索引 |
| 5 | paddle._C_ops.fmax | 逐元素比较取最大值（NaN处理） |
| 6 | paddle._C_ops.fmin | 逐元素比较取最小值（NaN处理） |
| 7 | paddle._C_ops.logsumexp | log空间的和（log-sum-exp技巧） |
| 8 | paddle._C_ops.max | 返回张量的最大值 |
| 9 | paddle._C_ops.maxout | Maxout激活（通道维度取最大） |
| 10 | paddle._C_ops.mean | 计算张量元素的平均值 |
| 11 | paddle._C_ops.prod | 计算张量元素的乘积 |
| 12 | paddle._C_ops.sum | 计算张量元素的和 |

### 统计量 (Statistics)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 13 | paddle._C_ops.nanmedian | 计算中位数（忽略NaN） |
| 14 | paddle._C_ops.variance | 计算方差 |

### 逻辑/计数归约 (Logical/Counting Reduction)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 15 | paddle._C_ops.all | 检查所有元素是否为真 |
| 16 | paddle._C_ops.any | 检查是否存在元素为真 |

### 累积归约 (Cumulative Reduction)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 17 | paddle._C_ops.cummax | 累积最大值 |
| 18 | paddle._C_ops.cummin | 累积最小值 |
| 19 | paddle._C_ops.cumprod | 累积乘积 |
| 20 | paddle._C_ops.cumsum | 累积和 |
| 21 | paddle._C_ops.logcumsumexp | log空间的累积和 |

### 归一化 (Normalization)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 22 | paddle._C_ops.frobenius_norm | 计算Frobenius范数 |
| 23 | paddle._C_ops.group_norm | 组归一化 |
| 24 | paddle._C_ops.instance_norm | 实例归一化 |
| 25 | paddle._C_ops.l1_norm | 计算L1范数 |
| 26 | paddle._C_ops.layer_norm | 层归一化 |
| 27 | paddle._C_ops.norm | 计算范数 |
| 28 | paddle._C_ops.p_norm | 计算p范数 |
| 29 | paddle._C_ops.renorm | 重新归一化 |
| 30 | paddle._C_ops.rms_norm | RMS归一化 |
| 31 | paddle._C_ops.squared_l2_norm | 计算L2范数的平方 |

### 池化归约 (Pooling Reduction)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 32 | paddle._C_ops.lp_pool2d | 2维Lp池化 |

### Softmax归约 (Softmax Reduction)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 33 | paddle._C_ops.log_softmax | Log Softmax激活函数 |

### 其他归约 (Other Reductions)

| 编号 | 算子 | 描述 |
|-----|------|------|
| 34 | paddle._C_ops.dist | 计算两个张量的距离 |
| 35 | paddle._C_ops.edit_distance | 编辑距离 |
| 36 | paddle._C_ops.numel | 返回张量元素总数 |
| 37 | paddle._C_ops.trace | 计算矩阵的迹（对角线元素和） |

## 总结

| 命名空间 | 归约算子数量 | 说明 |
|---------|------------|------|
| paddle.XXX | 40 | 最常用的高层API |
| paddle.nn.functional.XXX | 23 | 神经网络相关的函数式API |
| paddle._C_ops.XXX | 37+ | 底层C++算子接口 |

## 使用示例

```python
import paddle

# 1. paddle.XXX 示例
x = paddle.randn([3, 4, 5])

# 沿指定轴归约
mean_val = paddle.mean(x, axis=1)  # 结果形状: [3, 5]
sum_val = paddle.sum(x, axis=[1, 2])  # 结果形状: [3]
max_val = paddle.max(x)  # 全局最大值，标量

# 2. paddle.nn.functional.XXX 示例
x = paddle.randn([2, 3, 4, 4])

# 池化（空间维度归约）
pooled = paddle.nn.functional.avg_pool2d(x, kernel_size=2)  # 结果形状: [2, 3, 2, 2]

# Softmax（归一化归约）
softmax_out = paddle.nn.functional.softmax(x, axis=1)  # 沿通道维度归一化

# 3. paddle._C_ops.XXX 示例（通常不直接使用，但了解底层实现有帮助）
# 注意: _C_ops 是内部API，建议使用 paddle.XXX 或 paddle.nn.functional.XXX
```

## 归约操作的特点

1. **维度减少**: 归约操作会将张量的一个或多个维度减少
   - 例如: `[3, 4, 5]` → `[3, 4]` 或 `[]` (标量)

2. **沿轴计算**: 通常可以指定沿哪个或哪些轴进行归约
   - `paddle.sum(x, axis=1)`: 沿第1维求和
   - `paddle.sum(x, axis=[0, 2])`: 沿第0和第2维求和

3. **常见应用场景**:
   - 统计分析（均值、方差等）
   - 损失函数计算
   - 池化操作（特征降维）
   - 归一化（Batch Norm, Layer Norm等）
   - 注意力机制（Softmax）

## 注意事项

1. `paddle._C_ops` 中的算子是底层实现，不建议直接使用
2. `paddle.XXX` 和 `paddle.nn.functional.XXX` 通常会调用 `paddle._C_ops` 中的底层算子
3. 某些算子名称可能相似但行为略有不同，使用时请参考官方文档
4. 原地操作（以 `_` 结尾的算子）会直接修改输入张量，需谨慎使用

## 如何生成此文档

本文档使用 `tools/find_reduction_operators.py` 脚本自动生成，该脚本：
1. 解析 `python/paddle/__init__.py` 获取 paddle.XXX 的算子
2. 解析 `python/paddle/nn/functional/__init__.py` 获取 paddle.nn.functional.XXX 的算子
3. 解析 `paddle/phi/ops/yaml/ops.yaml` 获取 paddle._C_ops.XXX 的算子定义

## 更新日期

2025-10-31
