# 数学解码CNN：从骰子概率到MNIST实战，揭秘卷积神经网络的手写识别奥秘

> 本文通过骰子概率案例直观阐释卷积的数学本质，结合图像处理实例揭示卷积核提取空间特征的原理。基于PyTorch框架，逐步构建包含卷积层、ReLU激活和最大池化的CNN模型，在MNIST数据集上实现98.9%的识别准确率。通过对比全连接网络，详解卷积操作对局部特征提取的优越性，并剖析Adam优化器的双动量自适应学习机制。文章独创性地将概率论、线性代数与深度学习实践相融合，为读者提供"数学直觉-算法实现-性能优化"的三维认知路径。

> 在[上一篇文章](https://blog.csdn.net/galaxy223/article/details/146328910?fromshare=blogdetail&sharetype=blogdetail&sharerId=146328910&sharerefer=PC&sharesource=galaxy223&sharefrom=from_link)中，我们构建了一个单隐藏层的前馈全连接神经网络，完成了MNIST手写数字识别任务。本文将在原有模型基础上进行优化升级，重点引入图像处理领域的核心架构——卷积神经网络（Convolutional Neural Network, CNN）。

> 我的深度学习系列文章始终遵循三层认知路径：首先通过生活化案例搭建直觉认知框架；继而运用数学工具解析模型运作机理，在微积分与概率论层面锚定概念本质；最终才系统性构建专业术语体系——避免传统中式应试教学中前置概念灌输带来的认知负荷过载问题。

## 卷积是什么

> 这里推荐YouTube 3Blue1Brown的视频，可以非常直观的理解卷积，下文部分思路也参考自该视频

### 从投骰子到卷积

我们从一个经典概率问题开始：投掷两个骰子时，点数和的概率分布如何计算？传统解法需要穷举所有组合——比如和为4的情况有（1,3）、（2,2）、（3,1）三种，其概率即为
$\frac{1}{6}\times \frac{1}{6}\times 3=0.0833333$。看似简单？别急，让我们逐步增加难度。

现在假设每个骰子有20个面，问题似乎复杂一些，但是你依旧可以看出是1+19，2+18 …… 那么继续加大难度，每个骰子的质量并不均匀，每个面朝上的概率不在均等，这时你还固执地这样列式子那很容易出现计算错误或者思维混乱。聪明的同学会想到一个直观的办法，就像下面这样：
> 为了简单演示，我们仍然使用6面筛子，每个面朝上的概率不同

> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0505f9afc6aa47e69fabc9253073a5f4.png#pic_center)


如果你要找点数和为6出现的情况，那就是找下标和为6的格子，神奇的是，它们刚好在从右上到左下的一条直线上：

> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4b1ca22dbeeb45bcb8fc2844a5baa5f8.png#pic_center)


每个格子概率就是两个骰子对应点数朝上概率的乘积了(因为两个骰子的结果是相互独立的)。
你可能会说，你这有什么了不起的，不就是高中生物画的`9:3:3:1`的网格吗。别急，我们把列颠倒一下，像下面这样：

> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5490e8bfbe8a42a2b7178d4bc81380d2.png#pic_center)


发现没有，这下红色直线方向就变成了符合直觉感受的左上到右下了，可是如果我就想知道点数和6的情况，是不是图上太繁琐了一些，没关系，繁琐我们剔除掉就好：

> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a9d136005fda4ae5aa05fa98c80808b4.png#pic_center)


空格子怎么办呢？我们来做一个神奇的操作：

> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9b735fe6e83f4c70bb4f1fa1edb2e3c5.png#pic_center)


到这里就能看到卷积的雏形了，我们重点分析最后的图形，点数和为上的概率也就是
$$
P=p_{51}+p_{42}+p_{33}+p_{42}+p_{51}
$$
再比如如果你要求点数和为5的概率，那么就是下面这样：

> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/503c253009f8480790f80f9d1034f57a.png#pic_center)


求概率就变成了滑动下面的窗口了，是不是非常神奇。

接下来，换个情况，如果一个骰子只有三种情况(假设有这种骰子)，那么怎么用上面的“变换”求解呢？我们可以绘制一个简单的流程如下所示：
>![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6e09481e5b204df2b8affb3e25e5ab5b.png#pic_center)


我们只需要让小滑块在左右滑动即可
> 如果遇到边界情况就只算重叠部分

上述操作自然就引入概率论中的**卷积公式**：

对于离散型随机变量$X$和$Y$，其概率质量函数分别为$p_X(k)$和$p_Y(k)$，它们的和$Z=X+Y$的概率质量函数$p_Z(k)$为：
$$
p_Z(z)=\sum_kp_X(k)p_Y(z-k)
$$
其中求和遍历所有使$z-k$属于$Y$可能取值的$k$
> 本文只介绍了离散型随机变量的卷积公式，未考虑连续性随机变量，同时未对概率论中的核心定义定理做解释。由于本人精力、时间有限，后续可考虑通俗地讲解本科阶段各门数学科目。

### 卷积的初认识

上文我们直接给出了一个公式，说它叫卷积公式，但还是不知道什么是卷积，

卷积本质上是一种数学运算，我们从下面的一维卷积的例子开始理解：

假设有一个数组`[5, 2, 8, 1, 6, 3, 7, 4, 9, 0]`，它记录了一组数据(比如信号)，如下图所示：
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9bfa53aaef0a45bc8c6b770ecc58f486.png#pic_center)


这种数据的波动性太大，一个数据点完全可以远高于或远低于其他值，我们希望找到一种可以对其平滑化的办法，为此我们定义一个新的数组`[1/3,1/3,1/3]`，它有什么用呢？还记得我们之前算骰子概率的最后那一个图吗，我们的卷积操作和它几乎一模一样。
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e37001573f87444f8509f53d099ebaed.png#pic_center)


我们把三格大的小滑块放在索引`[3,4,5]`下面，我们模仿之前算概率时候的操作，`arr[5] * 1/3 + arr[4] * 1/3 + arr[3] * 1/3 = (3+6+1)/3=3.333`，我们将这个值作为一个新的数组(或者说变换后的信号)中间索引即`arr[4]`的值。类似地，我们可以计算出索引`0-9`的所有变换值，对于临界值，和概率的做法一样，把超出部分算作`0`即可。这样，我们得到了变换后的数组(信号)，如下图所示：
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/331d071b65004550bc264cd8720313ac.png#pic_center)


太神奇了！我们的信号不再有巨大波动了，而是一组看上去很和谐的数据，其实它的道理也很简单，试想一下，我们在计算红色数据的时候，我们每个索引对应的值不再仅取决于当前值，而是左右两个值和自己本身，至于每个值的影响，取决于对“滑动窗口”的设置，这个滑动窗口也就是**卷积核**，它充当了类似设置权重的作用，可以根据你想要提取的特征进行设置。上述的1/3均等设置是为了对数据做平滑处理，减小其波动性。
下面给出一维离散型的卷积运算定义：

对于两个离散序列$x[n]$和$h[n]$,其卷积运算定义为：
$$(y)[n]=(x*h)[n]=\sum_{k=-\infty}^{+\infty}x[k]\cdot h[n-k]$$
在实际应用中，若$x$和$h$分别为长度为$N$和$M$的有限序列，则求和范围被限制在有效区间内，结果序列$y$的长度为$N+M-1$。具体计算时，超出序列范围的项视为零。

有一点需要说明：上面图片中我们似乎并没有对卷积核做反转，但是在概率论里面我们似乎做了一次反转，但这并没有什么矛盾，因为我们上图给的只是反转后的滑动窗口，实际运算还是需要做一次类似反转的操作。就像下图一样
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9ed485ded2384bce9565db673d088d5b.png#pic_center)


我们按照图中虚线连接的一样将它们相乘后并累加，这就是翻转操作了。

> 需要特别说明的是，作者的专业背景聚焦于数学与计算机科学，对信号处理与电子工程相关的学科纵深（如滤波器设计原理）尚未建立系统认知，目前暂无学习计划。因此在讨论涉及滤波器等跨学科术语时，可能存在认知偏差，恳请予以指正补充。

### 二维卷积——图像处理

类似上述对信号做到处理，我们引入二维卷积操作，

假设我们有一张二维灰度图片(1为纯白色，0为纯黑色)，如下：
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b7c64a37fa5648c8ad1b700a658800c0.png#pic_center)


我们规定一个类似之前对一维信号处理类似的操作：如下图所示：
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/55a282bac1d942238e8e1ce3a6b6604b.png#pic_center)


上图中，`arr[5][6]`与淡蓝色3*3方格中心对齐，类似于对一维数组的操作，新的`arr[5][6]`值就是每一个对应方格相乘后的累加和1/3。我们来计算出所有位置变换后的值，如下：
> 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
>
> 0.333 0.333 0.333 0.333 0.333 0.333 0.000 0.000
>
> 0.333 0.333 0.333 0.333 0.333 0.333 0.000 0.000
>
> 0.333 0.333 0.333 0.333 0.333 0.333 0.000 0.000
>
> 0.333 0.667 1.000 1.000 0.667 0.333 0.000 0.000
>
> 0.000 0.000 0.000 0.333 0.333 0.333 0.000 0.000
>
> 0.000 0.000 0.000 0.333 0.333 0.333 0.000 0.000
>
> 0.000 0.000 0.000 0.333 0.333 0.333 0.000 0.000

我们把它可视化出来：
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2be481368aa2487b92ef58d23fbfcff6.png#pic_center)


你可能已经注意到特征图尺寸的缩减，这源于卷积核在滑动时未计算边缘区域的特性（数学上称为Valid卷积模式）。若需保持输入输出空间维度一致，可通过零填充（zero-padding）在原始图像外围扩展空白像素——当使用3×3卷积核时添加单像素边界，5×5核则需两像素边界，这正是卷积神经网络中Padding机制的核心作用。

观察卷积后的特征图，原始数字"4"的完整结构虽已模糊，但中心区域显现出显著的横向高亮带。这恰恰揭示了卷积操作的本质：通过空间滤波剥离表层信息，显影特定方向的边缘特征。值得注意的是，这个亮度突变的水平线正是数字4中部横笔画的数学表征。

在典型卷积神经网络架构中，单个卷积层往往配置32/64个独立卷积核并行运算，每个核作为独特的特征检测器，通过叠加多层卷积可逐步构建从边缘到纹理的分层特征抽象。

> 需要特别澄清一个关键差异点：我们在信号处理演示阶段执行的卷积核反转操作，在当前工程实现中已被刻意省略。这种取舍源于实际应用场景的效率考量(其实上面变换信号的过程也没必要反转卷积核)，原始的二维卷积数学表达的确需要有类似反转的操作，它的定义如下：

$$y[p,q]=\sum_m\sum_nx[m,n]\cdot h[p-m,q-n]$$

其中，$h[p-m,q-n]$表示翻转后的卷积核。

但实际应用中，我们使用如下公式：
$$y[p,q]=\sum_m\sum_nx[m,n]\cdot h[m+p,n+q]$$
或等价地(以更接近卷积的形式表示):
$$y[p,q]=\sum_m\sum_nx[m,n]\cdot h[m-p,n-q]$$


注意这里没有翻转操作，直接平移原核。

至此，相信你对卷积已经有了相当充分的理解

## 卷积神经网络

上述我们说，通过卷积运算，可以很好的提取图片的局部特征，而不再是孤立的看每一个像素点。这个思路可以用来改变我们上一篇文章中的全连接神经网络模型。
我们边写代码边理解流程


```python
import torch
from torchvision import transforms, datasets
from torch import nn
import torch.optim as optim

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)
```

当前数据预处理流程与上篇全连接网络的实现基本一致，但存在一个关键架构差异：我们刻意保留了`transforms`中**维度压缩操作**（即`x.view(-1)`）的缺失。这种设计选择源于神经网络结构的本质需求——卷积操作必须维持输入张量的空间拓扑信息（如28x28的二维结构），这与全连接网络强制展平像素为向量的处理方式形成鲜明对比。

张量展平操作并非被永久弃用，而是被精准延后至特征提取阶段之后。当卷积层完成空间特征提取任务，在接入全连接分类器之前，我们仍需要通过展平操作实现张量形态的转换，这正是分层特征处理范式的典型体现。

前面我们说过，卷积的作用是提取图片的局部特征，所以在整个模型流程图中，我们会做多次卷积操作以逐步提取它的特征。模型流程图如下所示：
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0df967979d154160ad745a7c901a3b34.png#pic_center)


刚开始你可能不理解为什么要做这些操作，比如为什么要做两次卷积不是三次？这些只能说是经过测试综合了效率与准确率后的一个折中较优的方案。你也可以自己定义你的模型。

我们先从一个例子来理解一下这个过程吧。


输入灰度手写数字7图片
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/89d0e0c3a5524e9ba26fd17845194130.png#pic_center)


我们选择16个卷积核，计算得出图片通过每一个卷积操作后的16张图片：
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7b30375bc1d043ebaa7ef4ce63afc77a.png#pic_center)

> 
> 这里需要注意，与传统图像处理手动定义卷积核参数不同，PyTorch框架采用随机初始化策略生成卷积核权重矩阵，通过反向传播动态优化参数，这种端到端特征学习范式正是深度学习区别于传统算法的核心优势。

这16张特征图可视化了不同卷积核对空间特征的差异化响应（如图示各通道激活模式）。为提取关键特征抑制噪声，我们首先对每个特征图施加ReLU激活函数。

接下来执行卷积网络的核心降维操作：池化(Pooling)。以最常用的2×2最大池化为例，该操作在局部窗口内保留最大值响应，如下图所示：
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d8db652a6dc2496fad6377b408694bf1.png#pic_center)


以2×2窗口为例，该操作对特征图实施网格化降采样：在每组相邻四个像素中保留最大值响应（对应图像中最显著的白亮特征），随后将四个像素单元压缩为单一像素单元。遍历全图执行该操作后，特征图分辨率精确缩减为原图的1/4（H/2 × W/2）。

现代网络设计中，部分架构使用步幅卷积(stride convolution)替代池化层，通过调节卷积核滑动步长实现降维，这种方案在保留空间信息完整性的同时达成分辨率控制

池化后的图片像下面这样：
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6992c40592c841f397babcf42292973b.png#pic_center)


接着重复上面的操作在经过卷积层，这次我们将特征数量(也就是卷积核数量)增加到32，经过卷积后的图像如下所示：
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/293026207ec040a1ae49071d917893a8.png#pic_center)


接着继续通过激活函数，再池化处理，如下图所示：
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8aca09515e9f4c30890a386efd87846c.png#pic_center)


到此为止，我们已经从单张图片几乎看不出来数字7了，但是它的特征几乎被榨干了。我们就可以通过全连接层，像我们上一篇文章一样，对其进行操作了。

我们来直接看代码吧：


```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.fc1 = nn.Linear(32 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = CNN()
```

尽管该类结构较为复杂，但只要掌握卷积网络的标准数据处理流，其设计逻辑便清晰可循：

1. **卷积层组件**  
   - `conv1/conv2`：采用PyTorch的`nn.Conv2d`标准接口  
     - `in_channels`：输入特征深度（首层为灰度图故设1）  
     - `out_channels`：输出特征图数量（即卷积核数量）  
     - `kernel_size`：空间感受野（卷积核大小）尺寸（如3×3）  
     - `padding`：输入边缘零填充像素数（维持分辨率）  
     - `stride`：卷积核滑动步长（默认为1，步长增大则特征图尺寸缩减）

2. **池化层策略**  
   - `pool`：实现2×2最大池化（MaxPool2d）  
     - 其他方案：平均池化（AvgPool2d）适用于平滑特征，最小池化（MinPool2d）用于异常检测  

3. **全连接层设计**  
   - `fc1/fc2`：继承自上一篇文章的全连接架构  
     - 参数选择：神经元数量属于超参数，需通过交叉验证调优

> 池化层输出为三维张量（通道×高×宽），接入全连接层前需执行`展平操作`（Flatten）转换为1D特征向量。该操作通过`x = x.view(-1, 16*5*5)`实现，其中16×5×5对应通道数与空间维度乘积。

后续操作与上一篇文章几乎完全一致，直接给出代码(可视化部分直接见上一篇文章)：


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 32

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {100 * correct / total:.2f}%")
```

    Epoch [1/10], Test Accuracy: 98.43%
    Epoch [2/10], Test Accuracy: 98.90%
    Epoch [3/10], Test Accuracy: 98.78%
    Epoch [4/10], Test Accuracy: 99.11%
    Epoch [5/10], Test Accuracy: 99.12%
    Epoch [6/10], Test Accuracy: 99.10%
    Epoch [7/10], Test Accuracy: 99.07%
    Epoch [8/10], Test Accuracy: 98.62%
    Epoch [9/10], Test Accuracy: 98.95%
    Epoch [10/10], Test Accuracy: 99.05%
    

可以看出，准确率提升明显

## 优化算法——Adam

> 本次CNN实现采用Adam优化器替代前作的全连接网络使用的SGD，该决策基于以下考量：SGD（随机梯度下降）需手动设置全局学习率，在CNN这种高维参数空间（通常含数万至百万级参数）中易陷入局部最优，收敛速度受学习率敏感度制约。

Adam（Adaptive Moment Estimation）是一种结合动量法和RMSProp的自适应学习率优化算法，广泛应用于深度学习。

理解Adam算法的关键在于把握其"动量追踪"与"梯度自适应"的双重机制，我们用登山者的运动学模型建立直觉认知：

1. **双动量系统建模**  
   - **一阶矩（动量方向）**：类比登山者运动惯性  
     > 当你在山坡持续朝右下方移动时，身体会积累向右的动量（历史梯度均值），即使遇到小坑洼（局部梯度突变），惯性仍会主导运动方向。数学上对应梯度方向的指数加权移动平均：
     
     $$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$  
   
   - **二阶矩（动量幅度）**：类比地表粗糙度感知  
     > 崎岖岩石区域需小步调整（高梯度方差时减小步长），平坦草地可跨步前进（低方差时增大步长）。数学表征为梯度平方的指数加权平均：
     
     $$v_t = \beta_2 v_{t-2} + (1-\beta_2)g_t^2$$
  
     其中，其中，$\beta_1$和$\beta_2$是衰减率(超参数)，$g$是当前位置的梯度大小，$t$是步数，初始为0，每一次迭代+1。

2. **参数更新动力学**  
   经过偏差校正（消除初始零偏置）：
   $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
   
   最终形成自适应学习步长：
   $$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
   - 分子$\hat{m}_t$：动量修正后的梯度方向  
   - 分母$\sqrt{\hat{v}_t}$：梯度幅度的自适应阻尼器  
   - $\eta$：全局学习率（方向盘）  
   - $\epsilon$：数值稳定项（防除零保护）

**算法特质**：这种"惯性导航+地形适应"机制，使得Adam在参数空间中既能保持动量加速，又能根据梯度地形智能调节步幅，完美平衡了SGD的探索性与RMSProp的自适应性。

## 后记

> 如果一个系统能够通过执行某个过程改进其性能这就是学习。——西蒙

至此，我们已完成卷积神经网络从理论构想到PyTorch实战的完整实现（输入28×28灰度图→特征提取→空间降维→分类决策）。该模型虽仅含两个卷积层（参数量约8.3K），却已充分展现CNN处理空间相关性的核心优势。

在后续的"深度学习"学习中，我们将从空间维度跨越至时序维度，探索循环神经网络（RNN）如何建立记忆单元捕捉序列依赖。届时将结合自然语言处理实例，揭示注意力机制如何模拟人类认知的聚焦特性。

愿这趟算法求真之旅，既能见证矩阵乘法中绽放的智能火花，亦能在梯度流动间感悟数理之美的永恒张力。前行路上，与君共勉。

Email: [wangk2829@gmail.com](mailto:wangk2829@gmail.com)
