# Extended Training Results - Research Paper

**Date**: 2025-10-12 07:55:10

**Total Experiments**: 6

## Summary Table

| Experiment                             | Dataset    | Model    | Task         |   Num Classes |   Epochs Trained |   Best Epoch |   Best Val Acc |   Test Acc |   Test Precision |   Test Recall |   Test F1 |   Training Time (min) |
|:---------------------------------------|:-----------|:---------|:-------------|--------------:|-----------------:|-------------:|---------------:|-----------:|-----------------:|--------------:|----------:|----------------------:|
| octmnist_advanced_50epochs_optimized   | octmnist   | advanced | single-label |             4 |               45 |           35 |         0.9232 |     0.725  |           0.7575 |        0.725  |    0.6978 |                255.59 |
| octmnist_simple_30epochs_optimized     | octmnist   | simple   | single-label |             4 |               30 |           25 |         0.9105 |     0.718  |           0.8039 |        0.718  |    0.6876 |                 68.67 |
| chestmnist_advanced_50epochs_optimized | chestmnist | advanced | multi-label  |            14 |               11 |            1 |         0.5419 |     0.5316 |           0.0001 |        0.0001 |    0.0001 |                 77.14 |
| dermamnist_advanced_50epochs_optimized | dermamnist | advanced | single-label |             7 |               18 |            8 |         0.7597 |     0.7357 |           0.7034 |        0.7357 |    0.7056 |                 28.53 |
| chestmnist_simple_30epochs_optimized   | chestmnist | simple   | multi-label  |            14 |               18 |            8 |         0.5419 |     0.5319 |           0.0005 |        0.0003 |    0.0004 |                 39.83 |
| dermamnist_simple_30epochs_optimized   | dermamnist | simple   | single-label |             7 |               30 |           29 |         0.7547 |     0.7332 |           0.6914 |        0.7332 |    0.6947 |                 48.3  |

## Best Results by Dataset

### CHESTMNIST

- **Best Model**: simple
- **Test Accuracy**: 0.5319
- **Test F1-Score**: 0.0004
- **Epochs Trained**: 18
- **Best Epoch**: 8
- **Training Time**: 39.83 minutes

### DERMAMNIST

- **Best Model**: advanced
- **Test Accuracy**: 0.7357
- **Test F1-Score**: 0.7056
- **Epochs Trained**: 18
- **Best Epoch**: 8
- **Training Time**: 28.53 minutes

### OCTMNIST

- **Best Model**: advanced
- **Test Accuracy**: 0.7250
- **Test F1-Score**: 0.6978
- **Epochs Trained**: 45
- **Best Epoch**: 35
- **Training Time**: 255.59 minutes

## Model Architecture Comparison

### Advanced

- **Mean Test Accuracy**: 0.6641 ± 0.0938
- **Best Accuracy**: 0.7357
- **Experiments**: 3

### Simple

- **Mean Test Accuracy**: 0.6610 ± 0.0915
- **Best Accuracy**: 0.7332
- **Experiments**: 3

