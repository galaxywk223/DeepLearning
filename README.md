# DeepLearning 学习笔记与实验

这个仓库整理了学习深度学习过程中的笔记（偏原理推导 + 直觉类比）以及可运行的 MNIST 实验代码。

## 导航

- 笔记：`notes/`
  - 多层感知机（MLP）手写数字识别：[notes/mlp-mnist.md](notes/mlp-mnist.md)
  - 卷积神经网络（CNN）数学直觉与 MNIST 实战：[notes/cnn-mnist.md](notes/cnn-mnist.md)
  - Transformer 自注意力机制推导：[notes/transformer-self-attention.md](notes/transformer-self-attention.md)
- 代码：`projects/mnist-cnn-experiments/`（见 [projects/mnist-cnn-experiments/README.md](projects/mnist-cnn-experiments/README.md)）
- 资源：`assets/`（图片/视频等）

## 演示视频

- 演示文件：`assets/videos/mnist-demo.mp4`
- 视频要点整理：`assets/videos/README.md`
- 备注：视频后半段的 CIFAR-10 图像分类与工程化优化部分，因重装系统导致源代码遗失；当前仓库主要保留可复现的 MNIST 实验与笔记。

## 快速运行（MNIST 实验）

在 `projects/mnist-cnn-experiments/` 目录下运行：

```bash
cd projects/mnist-cnn-experiments
pip install -r requirements.txt
python train_mlp.py
python train_cnn.py
```
