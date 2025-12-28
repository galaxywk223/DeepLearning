# MNIST Experiments

这个目录包含在学习过程中整理的 MNIST 训练脚本（MLP/CNN），用于把笔记里的推导落到可运行的代码上。

## 运行

```bash
pip install -r requirements.txt
python train_mlp.py
python train_cnn.py
```

## 说明

- 数据会自动下载到 `./data`（已在仓库中忽略）
- 训练过程的临时输出/权重文件默认不提交到仓库
