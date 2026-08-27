# Towards Trustworthy Dynamic Facial Expression Recognition via Information Bottleneck Modeling

The core implementations of the **Spatio-Temporal Modeling Module (STM)** and
the **Aware Calibration Loss (ACL)** in SAFE are made publicly available in
this repository.

## Abstract

Dynamic Facial Expression Recognition (DFER) requires robust temporal
representation learning under noisy video content, ambiguous expressions, and
imbalanced emotion categories. To address these challenges, SAFE introduces an
information-bottleneck-inspired framework for learning compact and trustworthy
video representations. This release contains two core components. First, the
Spatio-Temporal Modeling Module (STM) uses S2AM2 cells with sparse gating,
state-aware scaling, residual state fusion, lightweight self-attention, and
temporal aggregation to preserve informative dynamics while suppressing noisy
states. Second, the Aware Calibration Loss (ACL) jointly incorporates
confusion-aware calibration and supervised feature contrastive learning. ACL
emphasizes the most confusing non-target category, accounts for prediction
margin and category imbalance, and improves discriminative representation
learning and confidence calibration.

## Released Code

```text
SAFE_DFER/
├── STM.py    # Spatio-Temporal Modeling Module with S2AM2
└── ACL.py    # Aware Calibration Loss with CAL and FCL
```

### STM

`STM.py` contains:

- sliding-window temporal construction;
- sparse gating;
- state-aware scaling;
- residual state fusion;
- unidirectional and bidirectional S2AM2 layers;
- lightweight self-attention and temporal aggregation.

The module accepts clip-level features with shape `[B, T, D]`:

```python
from STM import STM

stm = STM(input_dim=512, hidden_dim=512, dropout=0.1, n_heads=8)
sequence_features, video_features, auxiliary = stm(clip_features)
```

### ACL

`ACL.py` contains the confusion-aware calibration loss (CAL), supervised
feature contrastive loss (FCL), and their joint SAFE objective:

```text
L = L_CE + lambda_cal * L_CAL + lambda_fcl * L_FCL
```

```python
from ACL import ACL

criterion = ACL(
    num_classes=7,
    lambda_cal=0.05,
    lambda_fcl=0.001,
    contrast_temperature=0.07,
)

loss, loss_items = criterion(
    logits,
    video_features,
    labels,
    class_counts=training_class_counts,
)
```

## Dependencies

```text
Python >= 3.9
PyTorch >= 1.12
```

## Acknowledgments

This project is built upon [DFEW](https://github.com/jiangxingxun/DFEW),
[FERV39k](https://github.com/wangyanckxx/FERV39k), and
[M3DFEL](https://github.com/Tencent/TFace/blob/master/attribute/M3DFEL/README.md).
Thanks to these excellent works!
