#!/usr/bin/env python3
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
"""
示例：如何使用 PaddlePaddle 归约操作算子

本示例展示了如何使用三个不同路径下的归约算子：
1. paddle.XXX
2. paddle.nn.functional.XXX
3. paddle._C_ops.XXX

注意：paddle._C_ops 是内部API，一般不建议直接使用
"""

# 注意：这是一个示例文件，展示归约操作的使用方式
# 如果要运行，需要先安装 PaddlePaddle

try:
    import paddle
    print("PaddlePaddle 已安装")
except ImportError:
    print("未安装 PaddlePaddle，这只是一个示例")
    print("请运行: pip install paddlepaddle 来安装")
    exit(0)

# 创建示例张量
print("=" * 60)
print("归约操作示例")
print("=" * 60)

# 创建一个 3D 张量
x = paddle.randn([2, 3, 4])
print(f"\n原始张量形状: {x.shape}")
print(f"原始张量:\n{x}")

# ============================================================
# 1. paddle.XXX 归约算子示例
# ============================================================
print("\n" + "=" * 60)
print("一、paddle.XXX 归约算子示例")
print("=" * 60)

# 统计归约
print("\n1.1 统计归约")
mean_val = paddle.mean(x)
print(f"paddle.mean(x): {mean_val.item():.4f} (全局平均值)")

mean_axis1 = paddle.mean(x, axis=1)
print(f"paddle.mean(x, axis=1) 形状: {mean_axis1.shape} (沿axis=1归约)")

sum_val = paddle.sum(x)
print(f"paddle.sum(x): {sum_val.item():.4f} (全局求和)")

max_val = paddle.max(x)
print(f"paddle.max(x): {max_val.item():.4f} (全局最大值)")

min_val = paddle.min(x)
print(f"paddle.min(x): {min_val.item():.4f} (全局最小值)")

# 索引归约
print("\n1.2 索引归约")
argmax_val = paddle.argmax(x)
print(f"paddle.argmax(x): {argmax_val.item()} (最大值的索引)")

# 统计量
print("\n1.3 统计量")
std_val = paddle.std(x)
print(f"paddle.std(x): {std_val.item():.4f} (标准差)")

var_val = paddle.var(x)
print(f"paddle.var(x): {var_val.item():.4f} (方差)")

# 逻辑归约
print("\n1.4 逻辑归约")
x_bool = paddle.to_tensor([[True, False], [True, True]])
all_val = paddle.all(x_bool)
print(f"paddle.all(x_bool): {all_val.item()} (是否所有元素为真)")

any_val = paddle.any(x_bool)
print(f"paddle.any(x_bool): {any_val.item()} (是否存在元素为真)")

# 累积归约
print("\n1.5 累积归约")
x_small = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
cumsum_val = paddle.cumsum(x_small)
print(f"paddle.cumsum([1,2,3,4]): {cumsum_val.numpy()}")

# ============================================================
# 2. paddle.nn.functional.XXX 归约算子示例
# ============================================================
print("\n" + "=" * 60)
print("二、paddle.nn.functional.XXX 归约算子示例")
print("=" * 60)

# Softmax归约
print("\n2.1 Softmax归约")
x_2d = paddle.randn([2, 5])
softmax_out = paddle.nn.functional.softmax(x_2d, axis=1)
print(f"原始张量形状: {x_2d.shape}")
print(f"paddle.nn.functional.softmax(x, axis=1) 形状: {softmax_out.shape}")
print(f"每行的和: {paddle.sum(softmax_out, axis=1).numpy()} (应该接近1)")

# 池化归约
print("\n2.2 池化归约")
x_4d = paddle.randn([1, 3, 8, 8])  # [batch, channel, height, width]
pooled = paddle.nn.functional.avg_pool2d(x_4d, kernel_size=2, stride=2)
print(f"原始张量形状: {x_4d.shape}")
print(f"avg_pool2d 后形状: {pooled.shape} (空间维度减半)")

# 归一化
print("\n2.3 归一化")
x_norm = paddle.randn([2, 3])
normalized = paddle.nn.functional.layer_norm(x_norm, x_norm.shape[1:])
print(f"原始张量:\n{x_norm.numpy()}")
print(f"layer_norm 后:\n{normalized.numpy()}")

# ============================================================
# 3. paddle._C_ops.XXX 归约算子（仅作了解，不建议直接使用）
# ============================================================
print("\n" + "=" * 60)
print("三、paddle._C_ops.XXX 归约算子（内部API）")
print("=" * 60)
print("\n注意：paddle._C_ops 是内部实现，通常通过 paddle.XXX 调用")
print("例如：paddle.mean() 内部会调用 paddle._C_ops.mean()")

# 说明
print("\n示例：")
print("- paddle.mean(x) -> 调用 -> paddle._C_ops.mean()")
print("- paddle.sum(x) -> 调用 -> paddle._C_ops.sum()")
print("- paddle.nn.functional.softmax(x) -> 调用 -> paddle._C_ops.softmax()")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
归约操作的特点：
1. 维度减少：将张量的一个或多个维度减少为更小维度或标量
2. 沿轴计算：可以指定沿哪个或哪些轴进行归约
3. 常见应用：统计分析、损失计算、池化、归一化等

三个命名空间：
1. paddle.XXX：最常用的高层API（推荐使用）
2. paddle.nn.functional.XXX：神经网络相关的函数式API（推荐使用）
3. paddle._C_ops.XXX：底层C++算子接口（内部使用，不推荐）

更多信息请查看：
- doc/REDUCTION_OPERATORS.md：完整的归约算子列表
- doc/REDUCTION_OPERATORS_README.md：工具使用说明
""")

print("\n运行成功！")
