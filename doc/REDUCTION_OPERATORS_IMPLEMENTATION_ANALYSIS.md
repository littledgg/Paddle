# Analysis of Reduction Operators Implementation

This table shows whether each reduction operator in paddle.XXX and paddle.nn.functional.XXX
calls paddle._C_ops for implementation.

## Summary Statistics

- **Total operators analyzed**: 63
- **Direct C++ calls** (_C_ops): 50 (79%)
- **Indirect calls** (through other paddle functions): 11 (17%)
- **Pure Python**: 2

## Detailed Analysis Table

| # | Operator | Code Path | Calls _C_ops | C_ops Used | In 37 Reduction List | Implementation Type | Confidence | Notes |
|---|----------|-----------|--------------|------------|---------------------|---------------------|------------|-------|
| 1 | `paddle.all` | python/paddle/tensor/math.py | ✅ Yes | all | all | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 2 | `paddle.amax` | python/paddle/tensor/math.py | ✅ Yes | amax | amax | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 3 | `paddle.amin` | python/paddle/tensor/math.py | ✅ Yes | amin | amin | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 4 | `paddle.any` | python/paddle/tensor/math.py | ✅ Yes | any | any | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 5 | `paddle.argmax` | python/paddle/tensor/search.py | ✅ Yes | argmax | argmax | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 6 | `paddle.argmin` | python/paddle/tensor/search.py | ✅ Yes | argmin | argmin | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 7 | `paddle.cartesian_prod` | python/paddle/tensor/math.py | ❌ No | - | - | Calls other paddle functions | ⭐⭐⭐ | Calls: to_tensor, meshgrid, cartesian... |
| 8 | `paddle.count_nonzero` | python/paddle/tensor/math.py | ❌ No | - | - | Calls other paddle functions | ⭐⭐⭐ | Calls: to_tensor, cast, count_nonzero |
| 9 | `paddle.cummax` | python/paddle/tensor/math.py | ✅ Yes | cummax | cummax | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 10 | `paddle.cummin` | python/paddle/tensor/math.py | ✅ Yes | cummin | cummin | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 11 | `paddle.cumprod` | python/paddle/tensor/math.py | ✅ Yes | cumprod | cumprod | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 12 | `paddle.cumprod_` | python/paddle/tensor/math.py | ✅ Yes | cumprod_ | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 13 | `paddle.cumsum` | python/paddle/tensor/math.py | ✅ Yes | cumsum | cumsum | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 14 | `paddle.cumsum_` | python/paddle/tensor/math.py | ✅ Yes | cumsum_ | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 15 | `paddle.dist` | python/paddle/tensor/linalg.py | ✅ Yes | dist | dist | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 16 | `paddle.einsum` | python/paddle/tensor/einsum.py | ❌ No | - | - | Calls other paddle functions | ⭐⭐⭐ | Calls: rand, seed, einsum |
| 17 | `paddle.fmax` | python/paddle/tensor/math.py | ✅ Yes | fmax | fmax | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 18 | `paddle.fmin` | python/paddle/tensor/math.py | ✅ Yes | fmin | fmin | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 19 | `paddle.logcumsumexp` | python/paddle/tensor/math.py | ✅ Yes | logcumsumexp | logcumsumexp | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 20 | `paddle.logsumexp` | python/paddle/tensor/math.py | ✅ Yes | logsumexp | logsumexp | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 21 | `paddle.max` | python/paddle/tensor/math.py | ✅ Yes | max | max | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 22 | `paddle.maximum` | python/paddle/tensor/math.py | ✅ Yes | maximum | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 23 | `paddle.mean` | python/paddle/tensor/stat.py | ✅ Yes | mean | mean | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 24 | `paddle.median` | python/paddle/tensor/stat.py | ❌ No | - | - | Pure Python | ⭐⭐⭐⭐ |  |
| 25 | `paddle.min` | python/paddle/tensor/math.py | ✅ Yes | min | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 26 | `paddle.minimum` | python/paddle/tensor/math.py | ✅ Yes | minimum | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 27 | `paddle.nanmean` | python/paddle/tensor/math.py | ❌ No | - | - | Calls other paddle functions | ⭐⭐⭐ | Calls: nanmean, isnan, divide |
| 28 | `paddle.nanmedian` | python/paddle/tensor/stat.py | ❌ No | - | - | Pure Python | ⭐⭐⭐⭐ |  |
| 29 | `paddle.nanquantile` | python/paddle/tensor/stat.py | ❌ No | - | - | Calls other paddle functions | ⭐⭐⭐ | Calls: to_tensor, full, nanquantile |
| 30 | `paddle.nansum` | python/paddle/tensor/math.py | ❌ No | - | - | Calls other paddle functions | ⭐⭐⭐ | Calls: to_tensor, zeros_like, nansum |
| 31 | `paddle.norm` | python/paddle/tensor/linalg.py | ❌ No | - | - | Calls other paddle functions | ⭐⭐⭐ | Calls: arange |
| 32 | `paddle.numel` | python/paddle/tensor/stat.py | ✅ Yes | numel | numel | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 33 | `paddle.prod` | python/paddle/tensor/math.py | ✅ Yes | prod | prod | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 34 | `paddle.quantile` | python/paddle/tensor/stat.py | ❌ No | - | - | Calls other paddle functions | ⭐⭐⭐ | Calls: arange, quantile |
| 35 | `paddle.renorm` | python/paddle/tensor/math.py | ✅ Yes | renorm | renorm | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 36 | `paddle.renorm_` | python/paddle/tensor/math.py | ✅ Yes | renorm_ | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 37 | `paddle.std` | python/paddle/tensor/stat.py | ❌ No | - | - | Calls other paddle functions | ⭐⭐⭐ | Calls: to_tensor, sqrt, std |
| 38 | `paddle.sum` | python/paddle/tensor/math.py | ✅ Yes | sum | sum | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 39 | `paddle.trace` | python/paddle/tensor/math.py | ✅ Yes | trace | trace | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 40 | `paddle.var` | python/paddle/tensor/stat.py | ❌ No | - | - | Calls other paddle functions | ⭐⭐⭐ | Calls: ones, cast, to_tensor |
| 41 | `paddle.nn.functional.adaptive_avg_pool1d` | python/paddle/nn/functional/pooling.py | ✅ Yes | pool2d | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 42 | `paddle.nn.functional.adaptive_avg_pool2d` | python/paddle/nn/functional/pooling.py | ✅ Yes | pool2d | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 43 | `paddle.nn.functional.adaptive_avg_pool3d` | python/paddle/nn/functional/pooling.py | ✅ Yes | pool3d | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 44 | `paddle.nn.functional.adaptive_max_pool1d` | python/paddle/nn/functional/pooling.py | ✅ Yes | max_pool2d_with_index | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 45 | `paddle.nn.functional.adaptive_max_pool2d` | python/paddle/nn/functional/pooling.py | ✅ Yes | max_pool2d_with_index | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 46 | `paddle.nn.functional.adaptive_max_pool3d` | python/paddle/nn/functional/pooling.py | ✅ Yes | max_pool3d_with_index | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 47 | `paddle.nn.functional.avg_pool1d` | python/paddle/nn/functional/pooling.py | ✅ Yes | pool2d | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 48 | `paddle.nn.functional.avg_pool2d` | python/paddle/nn/functional/pooling.py | ✅ Yes | pool2d | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 49 | `paddle.nn.functional.avg_pool3d` | python/paddle/nn/functional/pooling.py | ✅ Yes | pool3d | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 50 | `paddle.nn.functional.batch_norm` | python/paddle/nn/functional/norm.py | ✅ Yes | batch_norm, batch_norm_ | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 51 | `paddle.nn.functional.group_norm` | python/paddle/nn/functional/norm.py | ✅ Yes | group_norm | group_norm | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 52 | `paddle.nn.functional.instance_norm` | python/paddle/nn/functional/norm.py | ✅ Yes | instance_norm | instance_norm | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 53 | `paddle.nn.functional.layer_norm` | python/paddle/nn/functional/norm.py | ✅ Yes | layer_norm | layer_norm | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 54 | `paddle.nn.functional.local_response_norm` | python/paddle/nn/functional/norm.py | ❌ No | - | - | Calls other paddle functions | ⭐⭐⭐ | Calls: divide, scale, pow |
| 55 | `paddle.nn.functional.log_softmax` | python/paddle/nn/functional/activation.py | ✅ Yes | cast, log_softmax | log_softmax | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 56 | `paddle.nn.functional.lp_pool1d` | python/paddle/nn/functional/pooling.py | ✅ Yes | lp_pool2d | lp_pool2d | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 57 | `paddle.nn.functional.lp_pool2d` | python/paddle/nn/functional/pooling.py | ✅ Yes | lp_pool2d | lp_pool2d | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 58 | `paddle.nn.functional.max_pool1d` | python/paddle/nn/functional/pooling.py | ✅ Yes | max_pool2d_with_index, pool2d | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 59 | `paddle.nn.functional.max_pool2d` | python/paddle/nn/functional/pooling.py | ✅ Yes | max_pool2d_with_index, pool2d | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 60 | `paddle.nn.functional.max_pool3d` | python/paddle/nn/functional/pooling.py | ✅ Yes | max_pool3d_with_index, pool3d | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 61 | `paddle.nn.functional.maxout` | python/paddle/nn/functional/activation.py | ✅ Yes | maxout | maxout | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 62 | `paddle.nn.functional.softmax` | python/paddle/nn/functional/activation.py | ✅ Yes | cast, softmax | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |
| 63 | `paddle.nn.functional.softmax_` | python/paddle/nn/functional/activation.py | ✅ Yes | cast, softmax_ | - | Direct C++ call | ⭐⭐⭐⭐⭐ |  |

## Legend

- **Confidence Level**: ⭐ (1) = Low confidence, ⭐⭐⭐⭐⭐ (5) = High confidence
- **Calls _C_ops**: Whether the operator directly calls paddle._C_ops functions
- **C_ops Used**: Which _C_ops functions are called
- **In 37 Reduction List**: Which of the called _C_ops are in our list of 37 reduction operators
- **Implementation Type**:
  - `Direct C++ call`: Directly calls _C_ops
  - `Calls other paddle functions`: Calls other high-level paddle functions
  - `Pure Python`: Implemented in pure Python
  - `Unknown`: Could not determine
