# Look More but Care Less in Video Recognition

在视频识别任务中多看少管？

[TOC]



## 0、概述Abstract

Existing action recognition methods typically sample a few frames to represent each video to avoid the enormous computation, which often limits the recognition performance. 

现有的**动作识别**方法通常选取几帧来表示每个视频，以避免庞大的计算量，从而限制识别性能。

To tackle this problem, we propose Ample and Focal Network (AFNet), which is composed of two branches to utilize more frames but with less computation. 

为了解决这个问题，我们提出了由两个分支组成的**充足(Ample)**和**焦点(Focal)** 网络**(AFNet)**,以利用更多的帧，但计算量更少。

Specifically, the Ample Branch takes all input frames to obtain abundant information with condensed computation and provides the guidance for Focal Branch by the proposed Navigation Module; 

其中，**Ample分支**对所有输入帧进行压缩计算，获取丰富的信息，并通过所提出的导航模块为Facal分支提供指导。

the Focal Branch squeezes the temporal size to only focus on the salient frames at each convolution block; 

**Focal分支**压缩时间大小，只聚焦在每个卷积块的显著帧上；

in the end, the results of two branches are adaptively fused to prevent the loss of information.

最后，对两个分支的结果进行自适应融合，防止信息丢失。

 With this design, we can introduce more frames to the network but cost less computation. 

通过这种设计，我们可以在网络中**引入更多的帧，同时减少计算量。**

Besides, we demonstrate AFNet can utilize fewer frames while achieving higher accuracy as the dynamic selection in intermediate features enforces implicit temporal modeling. 

此外，我们证明AFNet可以使用更少的帧，同时获得更高的精度，因为中间特征的动态选择强制隐式时间建模。

Further, we show that our method can be extended to reduce spatial redundancy with even less cost. 

此外，我们证明了我们的方法可以扩展到以更低的成本减少空间冗余。

Extensive experiments on five datasets demonstrate the effectiveness and efficiency of our method. 

在5个数据集上的大量实验证明了该方法的有效性和高效性。

Our code is available at https://github.com/BeSpontaneous/AFNet-pytorch.

略。

## 1、介绍Introdution

### 1.1 第一段

Online videos have grown wildly in recent years and video analysis is necessary for many applications such as recommendation [6], surveillance [4, 5] and autonomous driving [33, 18]. 

近年来，网络视频迅猛发展，**视频分析**对于推荐、监控、自动驾驶等许多应用都是必不可少的。

These applications require not only accurate but also efficient video understanding algorithms.

这些应用需要准确并且高效的视频理解算法。

 With the introduction of deep learning networks [3] in video recognition, there has been rapid advancement in the performance of the methods in this area. 

随着深度学习网络在视频识别中的引入，该领域的方法的性能得到了快速的提升。

Though successful, these deep learning methods often cost huge computation, making them hard to be deployed in the real world.

虽然**这些深度学习方法**是成功的，但是它们通常需要**大量的计算**，这使得它们很难再现实世界中部署。

### 1.2 第二段

In video recognition, we need to sample multiple frames to represent each video which makes the computational cost scale proportionally to the number of sampled frames. 

在视频识别中，我们需要对多个帧进行采样来表示每个视频，这使得计算成本与采样帧的数量成正比。

In most cases, a small proportion of all the frames is sampled for each input, which only contains limited information of the original video. 

在大多数情况下，每个输入只对所有帧的一小部分进行采样，这些帧只包含原始视频的有限信息。

A straightforward solution is to sample more frames to the network but the computation expands proportionally to the number of sampled frames

一个直接的解决方案是向网络中采样更多的帧，但计算会随着采样帧的数量成比例地扩展。

### 1.3 第三段

There are some works proposed recently to dynamically sample salient frames [31, 17] for higher efficiency. 

为了提高效率，最近提出了一些动态采样显著帧的工作。

The selection step of these methods is made before the frames are sent to the classification network, which means the information of those unimportant frames is totally lost and it consumes a considerable time for the selection procedure.

这些方法的选择步骤都是在**帧发送到分类网络之前**进行的，这意味着那些不重要的帧的信息**完全丢失**，并且在选择过程中消耗了相当长的时间。

 Some other methods proposed to address the spatial redundancy in action recognition by adaptively resizing the resolution based on the importance of each frame [24], or cropping the most salient patch for every frame [30].

另外一些方法提出要解决动作识别中的空间冗余问题，如根据每帧的重要性自适应调整分辨率，或在每帧中裁剪最显著的patch。

 However, these methods still completely abandon the information that the network recognizes as unimportant and introduce a policy network to make decisions for each sample which leads to extra computation and complicates the training strategies.

然而，这些方法仍然完全放弃了网络识别为不重要的信息，并引入了策略网络对每个样本进行决策，这导致了额外的计算和复杂的训练策略。

![image-20231124215659318](.\\Look More but Care Less in Video Recognition.assets\image-20231124215659318.png)

#### 图1

Figure 1: Comparisons between existing methods and our proposed Ample and Focal Network (AFNet). 

图1：现有方法和我们提出的Ample和Focal网络(AFNet)之间的比较。

 Most existing works reduce the redundancy in data at the beginning of the deep networks which leads to the loss of information.

现有的大部分工作都是在深度网络的初始阶段减少数据冗余，从而导致信息的丢失。

We propose a two-branch design which processes frames with different computational resources within the network and preserves all input information as well.

我们提出了一种双分支的设计，该设计在网络中处理具有不同计算资源的帧，并保留所有输入信息。

### 1.4 第四段

In our work, we go from another perspective compared with previous works. 

在我们的工作中，与以往的作品相比，我们从另一个角度出发。

We propose a method which makes frame selection within the classification network. 

我们提出了一种在分类网络内进行帧选择的方法。

Shown in Figure 1, we design an architecture called Ample and Focal Network (AFNet) which is composed of two branches: 

如图1所示，我们设计了一个名为Ample和Facal网络(AFNet)的架构，有两个分支：

the ample branch takes a glimpse of all the input features with lightweight computation as we downsample the features for smaller resolution and further reduce the channel size;

Ample分支通过轻量级计算可以考虑所有输入特征，因为我们对特征进行下采样以获得更小的分辨率并进一步减小通道大小；

 the focal branch receives the guidance from the proposed navigation module to squeeze the temporal size by only computing on the selected frames to save cost;

Focal分支接收提供的导航模块的引导，仅对所选帧进行计算，压缩时间大小，节省成本；

 in the end, we adaptively fuse the features of these two branches to prevent the information loss of the unselected frames.

最后，我们自适应地融合了这两个分支的特征，以防止未选择帧的信息丢失。

### 1.5 第五段

In this manner, the two branches are both very lightweight and we enable AFNet to look broadly by sampling more frames and stay focused on the important information for less computation.

通过这种方式，两个分支都是非常轻量级的，我们使AFNet能够通过采样更多的帧来广泛地观察，并专注于重要的信息以减少计算。

Considering these two branches in a uniform manner, on the one hand, we can avoid the loss of information compared to other dynamic methods as the ample branch preserves the information of all the input;

统一考虑这两个分支，一方面，与其他动态方法相比，Ample分支保留了所有输入的信息，可以避免信息的丢失

 on the other hand, we can restrain the noise from the unimportant frames by deactivating them in each convolutional block. 

另一方面，我们可以通过在每个卷积块中去激活不重要帧来抑制噪声。

Further, we have demonstrated that the dynamic selection strategy at intermediate features is beneficial for temporal modeling as it implicitly implements frame-wise attention which can enable our network to utilize fewer frames while obtaining higher accuracy.

此外，我们已经证明了中间特征的动态选择策略有利于时间建模，因为它隐式地实现了逐帧注意，这可以使我们的网络在获得更高精度的同时使用更少的帧。

 In addition, instead of introducing a policy network to select frames, we design a lightweight navigation module which can be plugged into the network so that our method can easily be trained in an end-to-end fashion. 

此外，我们没有引入策略网络来选择框架，而是设计了一个轻量级的导航模块，可以插入到网络中，这样我们的方法就可以很容易地以端到端方式进行训练。

Furthermore, AFNet is compatible with spatial adaptive works which can help to further reduce the computations of our method

此外，AFNet兼容空间自适应工作，有助于进一步减少我们的方法的计算量。 

We summarize the main contributions as follows:

我们将其主要贡献总结如下：

• We propose an adaptive two-branch framework which enables 2D-CNNs to process more frames with less computational cost. With this design, we not only prevent the loss of information but strengthen the representation of essential frames.

我们提出了一种**自适应双分支框**架，使2D-CNN能够以**更少的计算成本处理更多帧**。

通过这种设计，我们不仅防止了信息的丢失，而且加强了基本框架的表示。

• We propose a lightweight navigation module to dynamically select salient frames at each convolution block which can easily be trained in an end-to-end fashion.

我们提出了一个**轻量级的导航模块**，可以在每个卷积块上**动态选择显著帧**，可以轻松地以**端到端**方式进行训练。

• The selection strategy at intermediate features not only empowers the model with strong flexibility as different frames will be selected at different layers, but also enforces implicit temporal modeling which enables AFNet to obtain higher accuracy with fewer frames.

中间特征的选择策略不仅使模型具有很强的灵活性，因为在不同的层上会选择不同的帧，而且还会强制执行隐式时间建模，使AFNet能够以更少的帧获得更高的精度。

• We have conducted comprehensive experiments on five video recognition datasets. The results show the superiority of AFNet compared to other competitive methods.

我们在5个视频识别数据集上进行了综合实验。结果表明，与其他竞争方法相比，AFNet具有优势。



## 2、 相关工作(Related Work)

### 2.1  视频识别Video Recognition

#### 2.1.1 第一段

The development of deep learning in recent years serves as a huge boost to the research of video recognition.

近年来深度学习的发展对视频识别的研究起到了巨大的推动作用。

 A straightforward method for this task is using 2D-CNNs to extract the features of sampled frames and use specific aggregation methods to model the temporal relationships across frames. 

一种简单的方法是使用2D-CNNs去提取采样帧的特征，并使用特定的聚合方法对帧间的时间关系进行建模

For instance, TSN [29] proposes to average the temporal information between frames. 

例如，TSN[29]提出对帧间的时间信息进行平均。

While TSM [21] shifts channels with adjacent frames to allow information exchange at temporal dimension.

而TSM[21]将信道与相邻帧进行移位，以实现时间维度上的信息交换。

Another approach is to build 3D-CNNs to for spatiotemporal learning, such as C3D [27], I3D [3] and SlowFast [9]. 

另一种方法是构建3D-CNNs来进行时空学习，如C3D，I3D和SlowFast[9]。

Though being shown effective, methods based on 3D-CNNs are computationally expensive, which brings great difficulty in real-world deployment.

虽然基于3D-CNNs的方法虽然被证明是有效的，但是计算成本很高，给实际部署带来了很大的困难。

#### 2.1.2 第二段

While the two-branch design has been explored by SlowFast, our motivation and detailed structure are different from it in the following ways: 

虽然SlowFast已经探索了双分支设计，但我们的动机和详细结构与它在以下方面有所不同：

1) network category: SlowFast is a static 3D model, but AFNet is a dynamic 2D network;

1）网络类别：SlowFast是静态3D模型，而AFNet是动态2D网络。

 2) motivation: SlowFast aims to collect semantic information and changing motion with branches at different temporal speeds for better performance, while AFNet is aimed to dynamically skip frames to save computation and the design of two-branch structure is to prevent the information loss; 

2）动机：SlowFast旨在收集语义信息，并在不同时间速度下改变分支的运动以获得更好的性能，

而AFNet旨在动态跳过帧以节省计算，双分支的设计是为了防止信息丢失。

3) specific design: AFNet is designed to downsample features for efficiency at ample branch while SlowFast processes features in the original resolution;

3）具体设计：AFNet旨在降低采样特征，以提高采样效率，而SlowFast在原始分辨率下处理特征；

 4) temporal modeling: SlowFast applies 3D convolutions for temporal modeling, AFNet is a 2D model which is enforced with implicit temporal modeling by the designed navigation module.

4）时间建模：SlowFast采用3D卷积进行时间建模，AFNet是一个2D模型，通过设计的导航模块进行隐式时间建模。

### 2.2 数据冗余Redundancy in Data

The efficiency of 2D-CNNs has been broadly studied in recent years.

近年来，2D-CNNs的效率得到了广泛的研究。

 While some of the works aim at designing efficient network structure [14], there is another line of research focusing on reducing the intrinsic redundancy in image-based data [34, 12]. 

虽然一些工作旨在设计高效的网络结构，但还有另一项研究侧重于减少基于图像的数据的内在冗余。

In video recognition, people usually sample limited number of frames to represent each video to prevent numerous computational costs.

在视频识别中，为了避免大量的计算成本，人们通常选取有限的帧来代表每个视频。

 Even though, the computation for video recognition is still a heavy burden for researchers and a common strategy to address this problem is reducing the temporal redundancy in videos as not all frames are essential to the final prediction. 

尽管如此，对研究人员来说，视频识别的计算仍然是个沉重的负担。，解决这个问题的一个常见的策略是减少视频中的时间冗余，因为并非所有帧对最终预测都是必不可少的。

[35] proposes to use reinforcement learning to skip frames for action detection. 

[35]提出使用强化学习跳过帧进行动作检测。

There are other works [31, 17] dynamically sampling salient frames to save computational cost.

还有其他工作[31,17]动态采样显著帧以节省计算成本

 As spatial redundancy widely exists in image-based data, [24] adaptively processes frames with different resolutions. 

由于图像数据中普遍存在空间冗余，[24]对不同分辨率的帧进行自适应处理。

[30] provides the solution as cropping the most salient patch for each frame.

[30]提供的解决方案是在每帧中裁剪最显著的patch。

However, the unselected regions or frames of these works are completely abandoned. 

然而，这些工作中未被选择的区域或框架被完全抛弃。

Hence, there will be some information lost in their designed procedures.

因此，在他们设计的过程中会有一些信息丢失。

 Moreover, most of these works adopt a policy network to make dynamic decisions, which introduces additional computation somehow and splits the training into several stages.

此外，这些工作大多采用策略网络进行动态决策，这以某种方式引入了额外的计算，并将训练分为几个阶段。

 In contrast, our method adopts a two-branch design, allocating different computational resources based on the importance of each frame and preventing the loss of information. 

相比之下，我们的方法采用双分支设计，根据每帧的重要性分配不同的计算资源，并防止信息丢失。

Besides, we design a lightweight navigation module to guide the network where to look, which can be incorporated into the backbone network and trained in an end-to-end way.

此外，我们设计了一个轻量级导航模块来引导网络去重视哪些，该模块可以并入骨干网络中，以端到端的方式进行训练。

 Moreover, we validate that the dynamic frame selection at intermediate features will not only empower the model with strong flexibility as different frames will be selected at different layers, but result in learned frame-wise weights which enforce implicit temporal modeling.

而且，我们验证了中间特征的动态帧选择不仅使模型具有很强的灵活性，因为在不同的层将选择不同的帧，而且还会产生学习到的帧加权，从而强制隐式时间建模。

## 3、方法Methodology

Intuitively, considering more frames enhances the temporal modeling but results in higher computational cost.

直观地说，考虑更多帧可以增强时间建模，但会导致更高的计算成本。

 To efficiently achieve the competitive performance, we propose AFNet to involve more frames but wisely extract information from them to keep the low computational cost.

为了有效地实现高性能，我们提出AFNet包含更多帧，但明智地从中提取信息以保持较低的计算成本。

 Specifically, we design a two-branch structure to treat frames differently based on their importance and process the features in an adaptive manner which can provide our method with strong flexibility. 

具体来说，我们设计了一种双分支结构，根据帧的重要性对其进行不同的处理，并以自适应的方式对特征进行处理，使我们的方法具有很强的灵活性。

Besides, we demonstrate that the dynamic selection of frames in the intermediate features results in learned frame-wise weights which can be regarded as implicit temporal modeling.

此外，我们还证明了在中间特征中动态选择帧会导致学习到帧权重，这可以被视为隐式时间建模。

### 3.1 体系结构设计Architecture Design

#### 图2 总体体系架构

![image-20231125101948584](.\\Look More but Care Less in Video Recognition.assets\image-20231125101948584.png)

Figure 2: Architecture of AF module. The module is composed of two branches, the ample branch would process all the input features in a lower resolution and reduced channel size;

图2：AF模块的体系结构。

该模块由两个支路组成，Ample分支将以较低的分辨率和减少的通道尺寸处理所有输入特征；

 while the focal branch would only compute the features of salient frames (colored features) guided by our proposed navigation module. 

而Focal分支将只计算由我们的导航模块引导的突出帧的特征(彩色特征)。

The results of two branches are adaptively fused at the end of AF module so that we can prevent the loss of information.

这两个分支的结果在AF模块的末端自适应融合，防止了信息的丢失。

#### 3.1.1  解释总体体系架构

As is shown in Figure 2, we design our Ample and Focal (AF) module as a two-branch structure: 

如图2所示，我们将Ample和Focal(AF)模块设计为两个分支结构：

the ample branch (top) processes abundant features of all the frames in a lower resolution and a squeezed channel size;

Ample分支(图2顶部)以较低的分辨率和压缩的通道大小处理所有帧的丰富特征；

 the focal branch (bottom) receives the guidance from ample branch generated by the navigation module and makes computation only on the selected frames. 

Focal分支(图2底部)接受导航模块生成的Ample分支的引导，只对选定的帧进行计算。

Such design can be conveniently applied to existing CNN structures to build AF module.

这样的设计可以方便地应用到现有的CNN结构中来构建AF模块。

#### 3.1.2  Ample分支

##### 3.1.2.1 第一段 介绍

**Ample Branch**. The ample branch is designed to involve all frames with cheap computation, which serves as 1) guidance to select salient frames to help focal branch to concentrate on important information; 2) a complementary stream with focal branch to prevent the information loss via a carefully designed fusion strategy

**Ample分支**.Ample分支设计成包含所有帧，计算量低。

1）可以指导Focal分支选择显著帧，帮助Focal分支集中于重要信息；

2）具有Focal分支的互补流，通过精心设计的融合策略防止信息丢失

##### 3.1.2.2 第二段 公式

Formally, we denote video sample $i$ as $v^i$, containing $T$ frames as $v^i=\left\{f_1^i,f_2^i,...,f_T^i\right\}$. 

正式地，我们将视频样本 $i$ 表示为 $v^i$, 其中包含 $T$ 帧，即  $v^i=\left\{f_1^i,f_2^i,...,f_T^i\right\}$。

For convenience, we omit the superscript $i$ in the following sections if no confusion arises. 

为了方便，在以下部分中，如果没有引起混淆，我们会省略上标 $i$ 。

 We denote the input of ample branch as $v_x\in\mathbb{R}^{T\times C\times H\times W}$, where $C$ represents the channel size and $H\times W$ is the spatial size.

我们将ample分支的输入表示为 $v_x\in\mathbb{R}^{T\times C\times H\times W}$,其中 $C$ 表示通道大小， $H\times W$ 是空间大小。

 The features generated by the ample branch can be written as:

由Ample分支生成的特征可以写成：

$v_{y^{a}}=F^{a}\left(v_{x}\right),\quad(1)$

where $v_{y^a}\in\mathbb{R}^{T\times(C_o/2)\times(H_o/2)\times(W_o/2)}$ represents the output of ample branch and $F^a$ stands for a series of convolution blocks. 

其中 $v_{y^a}\in\mathbb{R}^{T\times(C_o/2)\times(H_o/2)\times(W_o/2)}$ 代表Ample分支的输出， $F^a$ 表示一系列的卷积块。

While the channel, height, width at focal branch are denoted as $C_o,H_o$, $W_o$ correspondingly. 

而Focal分支的通道数、高度和宽度分别表示为 $C_o,H_o$, $W_o$ 。

We set the stride of the first convolution block to 2 to downsample the resolution of this branch and we upsample the feature at the end of this branch by nearest interpolation.

我们将第一个卷积块的步长设为2，以降低该分支的分辨率，并通过最近邻插值对该分支末端的特征进行上采样。

#### 3.1.3 导航模块Navigation Module

##### 3.1.3.1 第一段 介绍

**Navigation Module**. The proposed navigation module is designed to guide the focal branch where to look by adaptively selecting the most salient frames for video $v^i$ .

**导航模块**。所提出的导航模块旨在通过自适应地选择视频中最显著的帧来引导Focal分支注意哪里。

##### 3.1.3.2 第二段 公式

Specifically, the navigation module generates a binary temporal mask $L_n$ using the output from the $n$-th convolution block in ample branch $v_{y_n^a}$ . 

具体来说，导航模块使用Ample分支中第$n$个卷积块的输出 $v_{y_n^a}$生成一个二值化的时间掩码 $L_n$ 。

 At first, average pooling is applied to $v_{y_n^a}$ to resize the spatial dimension to $1\times1$, then we perform convolution to transform the channel size to 2:

首先，对 $v_{y_n^a}$ 进行平均池化以将空间维度调整为 $1\times1$，然后执行卷积操作将通道大小转换为2：

$\tilde{v}_{y_n^a}=\text{ReLU}\left(\text{BN}\left(W_1*\text{Pool}\left(v_{y_n^a}\right)\right)\right),(2)$

where * stands for convolution and $W_1$ denotes the weights of the $1\times1$ convolution. 

其中，*代表卷积运算， $W_1$ 表示 $1\times1$ 卷积的权重。

After that, we reshape the dimension of feature $\tilde{v}_{y_n^a}$ from $T\times2\times1\times1$ to $1\times(2\times T)\times1\times1$ so that we can model the temporal relations for each video from channel dimension by:

然后，我们将特征 $\tilde{v}_{y_n^a}$ 的维度从$T\times2\times1\times1$ 重塑为 $1\times(2\times T)\times1\times1$ ，以便我们可以从通道维度对每个视频建模时间关系通过：

$p_{n}^{t}=W_{2}*\tilde{v}_{y_{n}^{a}},(3)$

where $W_{2}$ represents the weights of the second $1\times1$ convolution and it will generate a binary logit $p_n^t\in\mathbb{R}^2$ for each frame $t$ which denotes whether to select it.

其中 $W_{2}$ 代表了第二个$1\times1$ 卷积的权重， 它将生成一个二值化的 $p_n^t\in\mathbb{R}^2$ ，用于表示第 $t$帧是否被选中。

However, directly sampling from such discrete distribution is non-differentiable.

然而，直接从这样的离散分布中抽样是不可微的。

 In this work, we apply Gumbel-Softmax [15] to resolve this non-differentiability. 

在这项工作中，我们使用Gumbel-Softmax [15] 来解决这种不可微

Specifically, we generate a normalized categorical distribution by using Softmax:

具体来说，我们使用Softmax生成一个归一化的分类分布：

$\pi=\left\{l_j\mid l_j=\frac{\exp\left(p_n^{t_j}\right)}{\exp\left(p_n^{t_0}\right)+\exp\left(p_n^{t_1}\right)}\right\},(5)$

and we draw discrete samples from the distribution π as:

我们从分布π中绘制离散样本为：

$L=\arg\max_j\left(logl_j+G_j\right),\quad(5)$

where $G_j=-\log(-\log U_j)$ is sampled from a Gumbel distribution and $U_j$ is sampled from Unif(0,1) which is a uniform distribution. 

其中$G_j=-\log(-\log U_j)$ 是从Gumbel分布中采样得到的，而 $U_j$ 是从 Unif(0,1) (均匀分布)中采样得到的。

As arg max cannot be differentiated, we relax the discrete sample $L$ in backpropagation via Sofunax:

由于无法对arg max进行微分，我们通过Sofumax在反向传播中对离散采样 $L$ 进行松弛处理。

$\hat{l_j}=\frac{\exp\left(\left(\log l_j+G_j\right)/\tau\right)}{\sum_{k=1}^2\exp\left(\left(\log l_k+G_k\right)/\tau\right)},(6)$

the distribution $\hat{l}$ will become a one-hot vector when the temperature factor $\tau\to0$ and we let $\tau$ decrease from 1 to 0.01 during training.

分布 $\hat{l}$ 将变成一个one-hot变量，当温度因子 $\tau\to0$ 时；我们在训练过程中让 $\tau$ 从 1逐渐减小至 0.01 。

#### 3.1.3  Focal分支

##### 3.1.3.1 第一段 介绍

**Focal Branch**. The focal branch is guided by the navigation module to only compute the selected frames, which diminishes the computational cost and potential noise from redundant frames.

**Focal 分支**。在导航模块的引导下，Focal分支只计算选定的帧，从而减少了计算成本和冗余帧的潜在噪声。

##### 3.1.3.2 第二段 公式

The features at the $n$-th convolution block in this branch can be denoted as 

$v_{y_n^f}\in\mathbb{R}^{T\times C_o\times H_o\times W_o}.$

在该分支中，第n个卷积块的特征可以表示为$v_{y_n^f}\in\mathbb{R}^{T\times C_o\times H_o\times W_o}.$

 Based on the temporal mask $L_n$ generated from the navigation module, we select frames which have corresponding non-zero values in the binary mask for each video and apply convolutional operations only on these extracted frames $v_{y_n^f}^{\prime}\in\mathbb{R}^{T_l\times C_o\times H_o\times W_o}:$

根据导航模块生成的时间掩码$L_n$,我们选择每个视频中在二值掩码中对应非零值的帧，并仅对这些提取的帧$v_{y_n^f}^{\prime}\in\mathbb{R}^{T_l\times C_o\times H_o\times W_o}:$进行卷积操作：

$v_{y_{n}^{f}}^{\prime}=F_{n}^{f}\begin{pmatrix}v'_{y_{n-1}^{f}}\end{pmatrix},(7)$

where $F_n^f$ is the $n$-th convolution blocks at this branch and we set the group number of convolutions to 2 in order to further reduce the computations. 

其中$F_n^f$是该分支中的第n个卷积块，并且我们设置了卷积操作的分组数为2，以进一步减少计算量。

After the convolution operation at $n$-th block, we generate a zero-tensor which shares the same shape with $v_{y_n^f}$ and fill the value by adding $v_{y_n^f}^{\prime}$ and  $v_{y_{n-1}^f}$ with the residual design following [13].

在第n个块的卷积操作之后，我们生成一个与 $v_{y_n^f}$ 形状相同的零张量，并通过残差设计将 $v_{y_n^f}^{\prime}$   和$v_{y_{n-1}^f}$相加来填充值。

At the end of these two branches, inspired by [1, 12], we generate a weighting factor $\theta$ by pooling and linear layers to fuse the features from two branches:

这两个分支的末端，受[1,12]启发，我们通过池化和线性层生成一个权重因子$\theta$，用来融合两个分支的特征。

$v_y=\theta\odot v_{y^a}+(1-\theta)\odot v_{y^f},(8)$

$\mathrm{where}\odot\text{denotes the channel-wise multiplication}.$

$\mathrm{其中}\odot\text{表示逐通道乘法}.$

### 3.1 隐式时间建模Implicit Temporal Modeling

While our work is mainly designed to reduce the computation in video recognition like[30, 25], we demonstrate that AFNet enforces implicit temporal modeling by the dynamic selection of frames in the intermediate features. 

虽然我们的工作主要旨在减少视频识别中的计算量，如[30,25]所示，但我们证明了AFNet通过动态选择中间特征的帧来实现隐式时间建模。

Considering a TSN[29] network which adapts vanilla ResNet[13] structure, the feature at the $n$-th convolutional block in each stage can be written as $v_n\in\mathbb{R}^{T\times C\times H\times W}.$ 

考虑一个适用于vanilla ResNet结构的TSN网络，在每个阶段的第n个卷积块处的特征可以表示为$v_n\in\mathbb{R}^{T\times C\times H\times W}$。

Thus, the feature at $n+1$-th block can be represented as:

因此，第n+1个块处的特征可以表示为：

$\begin{aligned}v_{n+1}&=v_n+F_{n+1}\left(v_n\right)\\&=\left(1+\Delta v_{n+1}\right)v_n,\end{aligned}(9)$

$\Delta v_{n+1}=\frac{F_{n+1}\left(v_n\right)}{v_n},(10)$

$\begin{aligned}
&\mathrm{where~}F_{n+1}\text{ is the }n+1\text{-th convolutional block and we define }\Delta v_{n+1}\text{ as the coefficient leamed from} \\
&\text{this block. By that we can write the output of this stage }v_N\text{ as:}
\end{aligned}$

在上述表达式中，$F_{n+1}$表示第n+1个卷积块，我们将$\Delta v_{n+1}$定义为从该块学习到的系数。通过这样的定义，我们可以将这个阶段的$v_N$表示为：

$v_N=\left[\prod_{n=2}^N\left(1+\Delta v_n\right)\right]*v_1.(11)$

Similarly, we define the features in ample and focal branch as:

同样，我们将ample和focal分支的特征定义为：

$\begin{gathered}
v_{y_{N}^{a}}=\left[\prod_{n=2}^{N}\left(1+\Delta v_{y_{n}^{a}}\right)\right]*v_{y_{1}},(12) \\
v_{y_{N}^{f}}=\left[\prod_{n=2}^{N}\left(1+L_{n}*\Delta v_{y_{n}^{f}}\right)\right]*v_{y_{1}}, (13)
\end{gathered}$





where $L_n$ is the binary temporal mask generated by Equation 5 and $v_{y_1}$ denotes the input of this Based on Equation 8, we can get the output of this stage as:

其中，$L_n$是由方程5生成的二元时间掩码，$v_{y1}$表示该阶段的输入。根据方程8，我们可以得到该阶段的输出。

$\begin{aligned}
v_{y_{N}}& =\theta\odot v_{y_{N}^{a}}+(1-\theta)\odot v_{y_{N}^{f}}  \\
&=\left\{\theta\odot\left[\prod_{n=2}^{N}\left(1+\Delta v_{y_{n}^{a}}\right)\right]+(1-\theta)\odot\left[\prod_{n=2}^{N}\left(1+L_{n}*\Delta v_{y_{n}^{f}}\right)\right]\right\}*v_{y_{1}}.& (14) 
\end{aligned}$



