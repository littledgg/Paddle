#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整列出 PaddlePaddle 三个路径下的所有归约操作算子

作者: PaddlePaddle Analysis Tool
日期: 2025-10-31
"""

import os
import ast
import yaml
from typing import List, Tuple

class ReductionOpFinder:
    """归约操作算子查找器"""
    
    def __init__(self):
        # 真正的归约操作：将张量的一个或多个维度减少为更小的维度
        self.true_reduction_ops = {
            # 基本统计归约
            'mean', 'sum', 'prod', 'max', 'min', 'amax', 'amin',
            
            # 带 NaN 处理的归约
            'nansum', 'nanmean', 'nanmedian', 'nanquantile',
            
            # 统计量
            'std', 'var', 'variance', 'median', 'quantile',
            
            # 逻辑归约
            'all', 'any',
            
            # 计数
            'numel', 'count_nonzero',
            
            # 索引归约
            'argmax', 'argmin',
            
            # 矩阵归约
            'trace', 'dist',
            
            # 累积归约（部分维度）
            'cumsum', 'cumprod', 'cummax', 'cummin',
            
            # Log空间归约
            'logsumexp', 'logcumsumexp',
            
            # 范数操作
            'norm', 'p_norm', 'l1_norm', 'frobenius_norm', 'squared_l2_norm',
            
            # 归一化操作（涉及统计量归约）
            'normalize', 'renorm',
            'layer_norm', 'batch_norm', 'group_norm', 'instance_norm',
            'local_response_norm', 'rms_norm',
            
            # 池化操作（空间归约）
            'avg_pool1d', 'avg_pool2d', 'avg_pool3d',
            'max_pool1d', 'max_pool2d', 'max_pool3d',
            'adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d',
            'adaptive_max_pool1d', 'adaptive_max_pool2d', 'adaptive_max_pool3d',
            'lp_pool1d', 'lp_pool2d',
            'maxout',
            
            # Softmax系列（归一化归约）
            'softmax', 'log_softmax',
            
            # 其他归约
            'einsum',  # 可以执行归约操作
            
            # 最小/最大的元素级操作（不完全是归约，但相关）
            'fmax', 'fmin', 'maximum', 'minimum',
            
            # 笛卡尔积
            'cartesian_prod',
        }
        
        # 需要排除的模式（虽然名字里有关键字，但不是归约）
        self.exclude_patterns = {
            'dataparallel', 'distributed', 'distribution', 'callbacks',
            'monkey_patch', 'start_api', 'summary', 'is_compiled',
            'normal', 'log_normal', 'standard_normal',
            'embedding_renorm', 'flash_', 'attention', 'attn',
            'fractional', 'unpool', 'triplet',
            'cdist', 'pdist', 'pairwise',
            'scaled_dot', 'gumbel',
            'adaptive_log_softmax_with_loss',
            'softmax_with_cross_entropy',
            'equal_all', 'allclose',
            'all_gather', 'all_reduce', 'all_to_all',
            'allreduce_sum', 'mp_allreduce_sum', 'c_allreduce_sum',
            'adamax', 'nadam', 
            'clip_by_norm', 'dgc_clip_by_norm',
            'fake_', 'dequantize_abs_max', 'quantize_abs_max',
            'fused_batch_norm_act', 'fused_rms_norm_ext',
            'fused_softmax_mask', 'sync_batch_norm',
            'spectral_norm', 'cross_entropy_with_softmax',
            'max_pool2d_with_index', 'max_pool3d_with_index',
            'ap_variadic', 'moe_gate', 'partial_',
            'mean_all',  # mean_all 是内部实现，通常使用 mean
        }
    
    def is_reduction_op(self, name: str) -> bool:
        """判断是否为归约操作"""
        name_lower = name.lower()
        
        # 先检查排除模式
        for exclude in self.exclude_patterns:
            if exclude in name_lower:
                return False
        
        # 检查是否在真正的归约操作集合中
        for op in self.true_reduction_ops:
            if op in name_lower:
                return True
        
        return False
    
    def extract_ops_from_init_file(self, filepath: str) -> List[str]:
        """从 __init__.py 提取导出的算子"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            ops = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name and alias.name != '*':
                            ops.append(alias.name)
            
            return ops
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return []
    
    def extract_ops_from_yaml(self, filepath: str) -> List[str]:
        """从 ops.yaml 提取算子定义"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            
            if not content:
                return []
            
            ops = []
            for item in content:
                if isinstance(item, dict) and 'op' in item:
                    ops.append(item['op'])
            
            return ops
        except Exception as e:
            print(f"Error reading yaml {filepath}: {e}")
            return []
    
    def get_op_category(self, op_name: str) -> str:
        """获取算子类别"""
        name_lower = op_name.lower()
        
        if 'pool' in name_lower:
            return '池化归约'
        elif 'norm' in name_lower:
            return '归一化'
        elif 'softmax' in name_lower:
            return 'Softmax归约'
        elif any(kw in name_lower for kw in ['cumsum', 'cumprod', 'cummax', 'cummin', 'logcumsum']):
            return '累积归约'
        elif any(kw in name_lower for kw in ['mean', 'sum', 'prod', 'max', 'min', 'amax', 'amin']):
            return '统计归约'
        elif any(kw in name_lower for kw in ['std', 'var', 'median', 'quantile']):
            return '统计量'
        elif any(kw in name_lower for kw in ['all', 'any', 'count']):
            return '逻辑/计数归约'
        elif any(kw in name_lower for kw in ['argmax', 'argmin']):
            return '索引归约'
        else:
            return '其他归约'
    
    def get_op_description(self, op_name: str) -> str:
        """获取算子描述"""
        descriptions = {
            # 基本统计归约
            'mean': '计算张量元素的平均值',
            'sum': '计算张量元素的和',
            'prod': '计算张量元素的乘积',
            'max': '返回张量的最大值',
            'min': '返回张量的最小值',
            'amax': '沿指定维度返回最大值',
            'amin': '沿指定维度返回最小值',
            'maximum': '逐元素比较取最大值',
            'minimum': '逐元素比较取最小值',
            'fmax': '逐元素比较取最大值（NaN处理）',
            'fmin': '逐元素比较取最小值（NaN处理）',
            
            # 逻辑归约
            'all': '检查所有元素是否为真',
            'any': '检查是否存在元素为真',
            
            # 统计量
            'std': '计算标准差',
            'var': '计算方差',
            'variance': '计算方差',
            'median': '计算中位数',
            'quantile': '计算分位数',
            'nanmedian': '计算中位数（忽略NaN）',
            'nanquantile': '计算分位数（忽略NaN）',
            'nansum': '计算和（忽略NaN）',
            'nanmean': '计算平均值（忽略NaN）',
            
            # 计数和追踪
            'numel': '返回张量元素总数',
            'count_nonzero': '计算非零元素数量',
            'trace': '计算矩阵的迹（对角线元素和）',
            'dist': '计算两个张量的距离',
            
            # 索引归约
            'argmax': '返回最大值的索引',
            'argmin': '返回最小值的索引',
            
            # 累积归约
            'cumsum': '累积和',
            'cumsum_': '累积和（原地操作）',
            'cumprod': '累积乘积',
            'cumprod_': '累积乘积（原地操作）',
            'cummax': '累积最大值',
            'cummin': '累积最小值',
            'logcumsumexp': 'log空间的累积和',
            'logsumexp': 'log空间的和（log-sum-exp技巧）',
            
            # 范数
            'norm': '计算范数',
            'p_norm': '计算p范数',
            'l1_norm': '计算L1范数',
            'frobenius_norm': '计算Frobenius范数',
            'squared_l2_norm': '计算L2范数的平方',
            'normalize': '归一化张量',
            'renorm': '重新归一化',
            'renorm_': '重新归一化（原地操作）',
            
            # 归一化层
            'layer_norm': '层归一化',
            'batch_norm': '批归一化',
            'group_norm': '组归一化',
            'instance_norm': '实例归一化',
            'local_response_norm': '局部响应归一化',
            'rms_norm': 'RMS归一化',
            
            # 池化
            'avg_pool1d': '1维平均池化',
            'avg_pool2d': '2维平均池化',
            'avg_pool3d': '3维平均池化',
            'max_pool1d': '1维最大池化',
            'max_pool2d': '2维最大池化',
            'max_pool3d': '3维最大池化',
            'adaptive_avg_pool1d': '1维自适应平均池化',
            'adaptive_avg_pool2d': '2维自适应平均池化',
            'adaptive_avg_pool3d': '3维自适应平均池化',
            'adaptive_max_pool1d': '1维自适应最大池化',
            'adaptive_max_pool2d': '2维自适应最大池化',
            'adaptive_max_pool3d': '3维自适应最大池化',
            'lp_pool1d': '1维Lp池化',
            'lp_pool2d': '2维Lp池化',
            'maxout': 'Maxout激活（通道维度取最大）',
            
            # Softmax系列
            'softmax': 'Softmax激活函数',
            'softmax_': 'Softmax激活函数（原地操作）',
            'log_softmax': 'Log Softmax激活函数',
            
            # 其他
            'einsum': '爱因斯坦求和约定（可执行归约）',
            'cartesian_prod': '笛卡尔积',
        }
        
        return descriptions.get(op_name, '归约操作算子')
    
    def format_output(self, ops: List[str], prefix: str, title: str, filepath: str) -> str:
        """格式化输出"""
        reduction_ops = [op for op in ops if self.is_reduction_op(op)]
        reduction_ops = sorted(set(reduction_ops))
        
        # 按类别分组
        categorized = {}
        for op in reduction_ops:
            category = self.get_op_category(op)
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(op)
        
        output = []
        output.append("=" * 80)
        output.append(title)
        output.append("=" * 80)
        output.append(f"代码路径: {filepath}")
        output.append(f"共找到 {len(reduction_ops)} 个归约算子")
        output.append("")
        
        # 按类别输出
        category_order = [
            '统计归约', '统计量', '逻辑/计数归约', '索引归约',
            '累积归约', '归一化', '池化归约', 'Softmax归约', '其他归约'
        ]
        
        idx = 1
        for category in category_order:
            if category in categorized:
                output.append(f"\n【{category}】")
                for op in sorted(categorized[category]):
                    desc = self.get_op_description(op)
                    output.append(f"  {idx:2d}. {prefix}.{op:30s} - {desc}")
                    idx += 1
        
        return "\n".join(output)

def main():
    paddle_root = "/home/runner/work/Paddle/Paddle"
    
    # 文件路径
    paddle_init = os.path.join(paddle_root, "python/paddle/__init__.py")
    nn_functional_init = os.path.join(paddle_root, "python/paddle/nn/functional/__init__.py")
    c_ops_file = os.path.join(paddle_root, "python/paddle/_C_ops.py")
    ops_yaml = os.path.join(paddle_root, "paddle/phi/ops/yaml/ops.yaml")
    
    finder = ReductionOpFinder()
    
    # 标题
    print("=" * 80)
    print("PaddlePaddle 归约操作算子完整列表")
    print("Complete List of Reduction Operators in PaddlePaddle")
    print("=" * 80)
    print()
    print("归约操作（Reduction Operations）是指将张量的一个或多个维度减少为")
    print("更小维度或标量的操作，例如求和、求平均、求最大值等。")
    print()
    
    # 1. paddle.XXX
    paddle_ops = finder.extract_ops_from_init_file(paddle_init)
    print(finder.format_output(paddle_ops, "paddle", 
                               "一、paddle.XXX 归约算子", paddle_init))
    print("\n")
    
    # 2. paddle.nn.functional.XXX
    nn_functional_ops = finder.extract_ops_from_init_file(nn_functional_init)
    print(finder.format_output(nn_functional_ops, "paddle.nn.functional",
                               "二、paddle.nn.functional.XXX 归约算子", 
                               nn_functional_init))
    print("\n")
    
    # 3. paddle._C_ops.XXX
    c_ops = finder.extract_ops_from_yaml(ops_yaml)
    print(finder.format_output(c_ops, "paddle._C_ops",
                               "三、paddle._C_ops.XXX 归约算子", 
                               f"{c_ops_file}\n算子定义: {ops_yaml}"))
    print("\n")
    
    # 额外说明
    print("=" * 80)
    print("说明")
    print("=" * 80)
    print()
    print("1. paddle._C_ops 模块的算子是从 C++ 动态加载的，来自:")
    print("   - paddle.base.core.eager.ops (动态图)")
    print("   - paddle.base.core.pir.ops (PIR)")
    print()
    print("2. paddle.XXX 和 paddle.nn.functional.XXX 中的许多算子")
    print("   内部会调用 paddle._C_ops 中的底层实现。")
    print()
    print("3. 归约操作的特点:")
    print("   - 减少张量的维度（例如: [3, 4, 5] -> [3, 4] 或 []）")
    print("   - 沿着一个或多个轴进行计算")
    print("   - 常见于统计、聚合和降维任务")
    print()
    print("4. 示例:")
    print("   - paddle.mean(x, axis=1)      # 沿axis=1归约")
    print("   - paddle.nn.functional.softmax(x, axis=-1)  # 归一化归约")
    print("   - paddle._C_ops.sum(x, ...)   # 底层C++实现")
    print()

if __name__ == "__main__":
    main()
