import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = driver.active.get_active_torch_device()

def is_hip():
    return driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() \
        and driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942', 'gfx90a', 'gfx908')


def naive_softmax(x: torch.Tensor):
    """Compute row-wise softmax of X using native pytorch

    We substract the maxium element in order to avoid overflows. Softmax is invariant to this shift.

    x  (M, N)
    │
    ├──→ x.max(dim=1)         → x_max    (M,)
    │                               │
    └──→ x - x_max[:, None]   → z        (M, N)
                                    │
                                torch.exp(z) → numerator  (M, N)
                                    │              │
                                    └── .sum(dim=1) → denominator (M,)
                                                            │
                                numerator / denominator[:, None] → ret (M, N)
    """
    # [:, None] 技巧总结
    # 等价写法：
    # x_max[:, None]        # shape (M,) → (M, 1)
    # x_max.unsqueeze(1)    # 同上, 在dim=1的维度上插入1个维度，相反的 squeeze(dim=1),就是删掉dim=1的这个维度，如果这一维度是1
    # x_max.reshape(-1, 1)  # 同上
    # 目的就是让 (M,) 变成 (M, 1)，这样才能和 (M, N) 做逐元素运算（广播规则要求维度对齐）。

    # read MM elements; write M elements
    # dim=k 表示消除第 k 个维度。shape=(M, N)，dim=1 消除 N → 剩下 (M,)，即每行变成一个标量。
    # x.max(dim=1) 返回一个 namedtuple(values, indices)
    x_max = x.max(dim=1)[0]
    # read MM + M elements; write MN elements
    # x_max shape: (M,) → x_max[:, None] shape: (M, 1) — 这是在最后加了一个维度
    # 广播机制：(M, N) - (M, 1) → 每行的每个元素都减去该行的最大值
    z = x - x_max[:, None]
    # read MN elements; write MN elements
    # 逐元素计算 e^z，shape 不变 (M, N)
    numerator = torch.exp(z)
    # read MM elements; write M elements
    # 按行求和
    # numerator shape: (M, N) → denominator shape: (M,)
    denominator = numerator.sum(dim=1)
    # read MN + M elements; write MN elements
    # 归一化
    # 同样的广播技巧：denominator[:, None] shape (M, 1)
    # 结果每行加起来 = 1，就是概率分布
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements; wrote 3MNS + 2M elements
    return ret

x = torch.rand((4, 3))
ret =naive_softmax(x)

print(f"{x=}")
print(f"{ret=}")
