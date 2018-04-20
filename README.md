# SlimNets

A comparison of popular methods to create more efficient (smaller and faster) Nueral Networks (post training, or during training?).


Framework:
python-tensorflow for GCP compatibillity

Dataset:
CIFAR10

Model: Smallish models
MobileNet, VGGnet

Methods:
1.Pruning
2.Low-Rank Factorization
3.Knowledge Distillation

Absolute Metrics: create framework for extracting and visualizing these quantities from various methods

1. Model size (MB)
2. Test Accuracy
3. Training Time (h)
4. Inference Time (s)

Relative Metrics:
1. Model compression rate
2. training/inference accelerations
3. relative accuracy between prunned models and models with 

