# OWOD学习

> **参考博客**：
>
> - http://t.csdnimg.cn/gOsze
> -  https://zhuanlan.zhihu.com/p/386046811
>
> **论文地址：**https://arxiv.org/abs/2103.02603
>
> **代码地址**：[GitHub - JosephKJ/OWOD: (CVPR 2021 Oral) Open World Object Detection](https://github.com/JosephKJ/OWOD)



**目录：**

[TOC]

## 0 摘要

Humans have a natural instinct to identify unknown object instances in their environments. The intrinsic curiosity about these unknown instances aids in learning about them, when the corresponding knowledge is eventually available. This motivates us to propose a novel computer vision problem called: ‘Open World Object Detection’, where a model is tasked to:<font color="red"> 1) identify objects that have not been introduced to it as ‘unknown’, without explicit supervision to do so, and 2) incrementally learn these identified unknown categories without forgetting previously learned classes, when the corresponding labels are progressively received. </font>We formulate the problem, introduce a strong evaluation protocol and provide a novel solution, which we call ORE: Open World Object Detector, based on <font color="blue">contrastive clustering </font>and <font color="blue">energy based unknown identification</font>. Our experimental evaluation and ablation studies analyse the efficacy of ORE in achieving Open World objectives. As an interesting by-product, we find that identifying and characterizing unknown instances helps to reduce confusion in an incremental object detection setting, where we achieve state-of the-art performance, with no extra methodological effort. We hope that our work will attract further research into this newly identified, yet crucial research direction.
**人类有识别环境中未知物体实例的本能。当相应的知识最终可用时，对这些未知物体实例的好奇心有助于了解它们。这促使我们提出了一个新的计算机视觉问题，称为：ORE，或者叫"开放世界目标检测(Open World Object Detection)"，模型的任务是：<font color="red">1) 在没有明确监督的情况下，将没有被引入的物体识别为“未知物体”，2）当逐渐接收到相应的标签时，增量地学习这些识别出的未知类别而不忘记先前学习的类。</font>本文提出了一种基于<font color="blue">对比聚类</font>和<font color="blue">基于能量的未知识别</font>的开放世界目标检测算法。我们的实验评估和消融研究分析了ORE网络在实现开放世界目标方面的功效。作为一个有趣的研究结果，我们发现识别和描述未知实例有助于减少增量目标检测设置中的混乱，在这种情况下，我们实现了最先进的性能，而无需额外的方法学习。我们希望，我们的工作将吸引进一步研究这个新颖但至关重要的研究方向。**

## 1 介绍

### 1-1 开放世界目标检测

Deep learning has accelerated progress in Object Detection research [13, 53, 18, 30, 51], where a model is tasked to identify and localise objects in an image. All existing approaches work under a strong assumption that all the classes that are to be detected would be available at training phase. <font color="red">Two challenging scenarios arises when we relax this assumption: 1) A test image might contain objects from unknown classes, which should be classified as unknown. 2) As and when information (labels) about such identified unknowns become available, the model should be able to incrementally learn the new class. </font>Research in developmental psychology [40, 35] finds out that the ability to identify what one doesn’t know, is key in captivating curiosity. Such a curiosity fuels the desire to learn new things [8, 15]. This motivates us to propose a new problem where a model should be able to identify instances of unknown objects as unknown and subsequently learns to recognise them when training data progressively arrives, in a unified way. We call this problem setting as Open World Object Detection.
**深度学习加速了目标检测研究的进展[13,53,18,30,51]，模型的任务是识别和定位图像中的目标。所有现有的方法都是在一个重要假设下工作的，即：所有要检测的类在训练阶段都是可用的。<font color="red">当我们放宽这一假设时，出现了两个具有挑战性的场景：1）测试图像可能包含来自未知类的目标，这些目标应该被分类为未知。2）当有关这些已识别未知项的信息（标签）可用时，模型应该能够增量地学习新类。</font>发展心理学的研究[40,35]发现，辨别自己不知道的东西的能力是吸引好奇心的关键。这种好奇心激发了人们学习新事物的欲望[8,15]。这促使我们提出一个新的问题，即模型应该能够将未知的实例识别为未知目标，然后在训练数据以统一的方式逐渐到达时识别它们。我们把这个问题称为开放世界目标检测。**

### 1-2 开放集分类和开放世界分类

The number of classes that are annotated in standard vision datasets like Pascal VOC [9] and MS-COCO [31] are very low (20 and 80 respectively) when compared to the infinite number of classes that are present in the open world. <font color="red">Recognising an unknown as an unknown requires strong generalization. Scheirer et al. [56] formalises this as **Open Set classification** problem.</font> Henceforth, various methodologies (using 1-vs-rest SVMs and deep learning models) has been formulated to address this challenging setting. <font color="red">Bendale et al. [2] extends Open Set to an **Open World classification** setting by additionally updating the image classifier to recognise the identified new unknown classes. </font>Interestingly, as seen in Fig. 1, Open World object detection is unexplored, owing to the difficulty of the problem setting.
**与开放世界中存在的无限数量的类相比，标准视觉数据集（如Pascal VOC[9]和MS-COCO[31]）中注释的类的数量非常少（分别是20和80）。<font color="#ff9580">将未知目标识别为"未知" 需要很强的泛化能力。Scheirer等人[56]将其形式化为</font><font color="red">开放集分类</font><font color="#ff9580">问题</font>。从今以后，各种方法（使用1-vs-rest支持向量机和深度学习模型）都被用来解决这一具有挑战性的问题。<font color="#ff9580">Bendale等人[2]通过额外更新图像分类器来识别新未知类别，将开放集扩展到</font><font color="red">开放世界分类</font><font color="#ff9580">设置。</font>有趣的是，如图1所示，由于问题的困难，开放世界目标检测还未被探索。**

:grey_question: ![image-20231030215829431](F:\论文\毕设学习\OWOD.assets\image-20231030215829431.png)

:grey_question:参考：http://t.csdnimg.cn/IwSuA

##### 图1：三个轴: 增量学习vs开集学习vs问题难度

:grey_question:解释：

:grey_question:1、增量学习轴：增量学习的能力

:grey_question:2、开集学习：开放集->开放世界

:grey_question:3、问题难度：图像分类->图像检测(物体识别)

![image-20231030151659455](F:\论文\毕设学习\OWOD.assets\image-20231030151659455.png)

> Figure 1: Open World Object Detection(:star:) is a novel problem that has not been formally defined and addressed so far. Though related to the Open Set and Open World classification, Open World Object Detection offers its own unique challenges, which when addressed, improves the practicality of object detectors. 
>
> **图1：开放世界目标检测(:star:)是一个到目前为止还没有正式定义和解决的新问题。虽然与开放集和开放世界分类相关，但开放世界目标检测有其独特的挑战性，当解决这些问题时，提高了目标检测器的实用性**

### 2-3 开放集分类、开放世界分类和开放世界目标检测

The advances in Open Set and Open World image classification cannot be trivially adapted to Open Set and Open World object detection, because of a fundamental difference in the problem setting: The object detector is trained to detect unknown objects as background. Instances of many unknown classes would have been already introduced to the object detector along with known objects. As they are not labelled, these unknown instances would be explicitly learned as background, while training the detection model. Dhamija et al. [7] finds that even with this extra training signal, the state-of-the-art object detectors results in false positive detections, where the unknown objects end up being classified as one of the known classes, often with very high probability. Miller et al. [42] proposes to use dropout sampling to get an estimate of the uncertainty of the object detection prediction. This is the only peer-reviewed research work in the open set object detection literature. Our proposed Open World Object Detection goes a step further to incrementally learn the new classes, once they are detected as unknown and an oracle provides labels for the objects of interest among all the unknowns. To the best of our knowledge this has not been tried in the literature. Fig. 1 shows a taxonomy of existing research work in this space
**开放集和开放世界图像分类的进展不能简单地用于开放集和开放世界的目标检测，因为问题设置有一个根本的区别：目标检测器被训练来检测未知目标作为背景。许多未知类的实例已经与已知目标一起引入目标检测器中。由于没有标记，这些未知实例将被显式地学习为背景，同时训练检测模型。Dhamija等人[7]发现，即使有了这个额外的训练信号，最先进的物体检测器也会产生假阳性检测，其中未知物体最终被归类为已知类别之一，通常概率非常高。Miller等人[42]建议使用脱落采样来估计目标检测预测的不确定性。这是开放集目标检测文献中唯一一项同行评议的研究工作。我们提出的开放世界目标检测方法更近一步，一旦新类被检测为未知，并且预测为所有未知目标中感兴趣的目标提供标签，就可以增量地学习它们。据我们所知，这在文献中还没有尝试过。图1显示了该领域现有研究工作的分类。**

### 2-4 开放世界目标检测目标

The Open World Object Detection setting is much more natural than the existing closed-world, static-learning setting. The world is diverse and dynamic in the number, type and configurations of novel classes. It would be naive to assume that all the classes to expect at inference are seen during training. Practical deployments of detection systems in robotics, self-driving cars, plant phenotyping, healthcare and surveillance cannot afford to have complete knowledge on what classes to expect at inference time, while being trained in-house. The most natural and realistic behavior that one can expect from an object detection algorithm deployed in such settings would be to confidently <font color="red">predict an unknown object as unknown, and known objects into the corresponding classes. As and when more information about the identified unknown classes becomes available, the system should be able to incorporate them into its existing knowledge base. </font>This would define a smart object detection system, and ours is an effort towards achieving this goal.

**与现有的封闭世界静态学习设置相比，开放世界目标检测设置更加自然。世界在新类上的数量、类型和结构上是多样化和动态的。假设所有在推理时预期的类都是在训练期间看到的，这是不现实的。在机器人、自动驾驶汽车、植物表型鉴定、医疗保健和监控等领域，检测系统的实际部署不能完全掌握推理时需要学习的类别，同时还要接受内部训练。在这样的环境中部署的目标算法最自然、最现实的行为是<font color="red">将未知实例预测为未知目标，并将已知实例划分为相应的类。当更多关于已识别未知类的信息可用时，系统应该能够将它们合并到现有的知识库中。</font>这将定义一个只能目标检测系统，我们正努力实现这一目标。**

### 2-5 本论文贡献

The key contributions of our work are:

-  We introduce a novel problem setting, Open World Object Detection, which models the real-world more closely.
- We develop a novel methodology, called <font color="red">ORE, based on contrastive clustering, an unknown-aware proposal network and energy based unknown identification </font>to address the challenges of open world detection.
- We introduce a comprehensive experimental setting, which helps to measure the open world characteristics of an object detector, and benchmark ORE on it against competitive baseline methods.
-  As an interesting by-product, the proposed methodology achieves state-of-the-art performance on Incremental Object Detection, even though not primarily designed for it

**我们的工作做出的主要贡献有：**

- **我们引入了一种新的问题设置，即开放世界目标检测，它可以更紧密地模拟现实世界。**
- **我们开发了一种新的方法，称为<font color="red">ORE，基于对比聚类、未知-感知建议网络和基于能量的未知识别</font>来应对开放世界检测的挑战。**
- **我们引入了一个全面的实验环境，有助于测量目标检测器的开放世界特性，并将ORE与竞争性基线方法进行比较。**
- **作为一个有趣的副产品，所提出的方法在增量目标检测方面实现了最先进的性能，尽管主要不是为其设计的。**

## 2 相关工作

### 2-1 开放集分类

Open Set Classification: The open set setting considers knowledge acquired through training set to be incomplete, thus new unknown classes can be encountered during testing. Scheirer et al. [57] developed open set classifiers in a one-vs-rest setting to balance the performance and the risk of labeling a sample far from the known training examples (termed as open space risk). Follow up works [22, 58] extended the open set framework to multi-class classifier setting with probabilistic models to account for the fading away classifier confidences in case of unknown classes.
**开放集分类：开放集设置认为通过训练集获得的知识是不完整的，因此在测试过程中会遇到新的未知类。Scheirer等人[57]在一对一的环境中开发了开放集分类器，以平衡标记远离已知训练示例的样本的性能和风险（称为开放空间风险）。后续工作[22,58]将开放集框架扩展到多类分类器设置，并采用概率模型来解释未知类情况下分类器置信度的衰减**

Bendale and Boult [3] identified unknowns in the feature space of deep networks and used a Weibull distribution to estimate the set risk (called OpenMax classifier). A generative version of OpenMax was proposed in [12] by synthesizing novel class images. Liu et al. [34] considered a long-tailed recognition setting where majority, minority and unknown classes coexist. They developed a metric learning framework identify unseen classes as unknown. In similar spirit, several dedicated approaches target on detecting the out of distribution samples [29] or novelties [47]. Recently, self-supervised learning [45] and unsupervised learning with reconstruction [64] have been explored for open set recognition. However, while these works can recognize unknown instances, they cannot dynamically update themselves in an incremental fashion over multiple training episodes. Further, our energy based unknown detection approach has not been explored before.
**Bendale和Boult[3]在深度网络的特征空间中识别未知目标，并使用Weibull分布来估计集合风险（称为OpenMax分类器）。[12]通过合成新的类图像，提出了OpenMax的进化版本。Lin等人[34]考虑了一种长-尾识别环境，其中多数类、少数类和未知类共存。他们开发了一个度量学习框架，将看不见的类识别为未知类。本着类似的精神，有几种专门的方法旨在检测分布外的样本[29]或新类别[47]。最近，自监督学习[45]和带重构的无监督学习[64]被探索用于开放集识别。然而，虽然这些工作可以识别未知的实例，但它们不能在多个训练集上以增量方式动态更新自己。此外，我们基于能量的未知检测方法还没有被探索过。**

### 2-2 开放世界分类 

Open World Classification: [2] first proposed the open world setting for image recognition. Instead of a static classifier trained on a fixed set of classes, they proposed a more flexible setting where knowns and unknowns both coexist. The model can recognize both types of objects and adaptively improve itself when new labels for unknown are provided. Their approach extends Nearest Class Mean classifier to operate in an open world setting by re-calibrating the class probabilities to balance open space risk. [46] studies open world face identity learning while [63] proposed to use an exemplar set of seen classes to match them against a new sample, and rejects it in case of a low match with all previously known classes. However, they don’t test on image classification benchmarks and study product classification in e-commerce applications.
**开放世界分类：[2]首先提出了图像识别的开放世界设置。他们提出了一种更灵活的设置，即已知和未知同时存在，而不是在一组固定的类上训练静态分类器。该模型能同时识别这两种类型的目标，并在为未知目标提供新的标签时自适应地进行改进。他们的方法通过重新校准类概率来平衡开放空间风险，从而扩展了最近类均值分类器，使其在开放世界环境中运行。[46]研究了开放世界的人脸识别学习，而[63]则建议使用一组已知类的样本来匹配新样本，如果与所有已知类的匹配度较低，则拒绝使用。然而，他们并没有对图像分类基准进行测试，也没有研究电子商务应用中的产品分类。**

### 2-3 开放集检测

Open Set Detection: Dhamija et al. [7] formally studied the impact of open set setting on popular object detectors. They noticed that the state of the art object detectors often classify unknown classes with high confidence to seen classes. This is despite the fact that the detectors are explicitly trained with a background class [54, 13, 32] and/or apply one-vs-rest classifiers to model each class [14, 30]. A dedicated body of work [42, 41, 16] focuses on developing measures of (spatial and semantic) uncertainty in object detectors to reject unknown classes. E.g., [42, 41] uses Monte Carlo Dropout [11] sampling in a SSD detector to obtain uncertainty estimates. These methods, however, cannot incrementally adapt their knowledge in a dynamic world.
**开放集检测：Dhamija等人[7]正式研究了开放集设置对流行目标检测器的影响。他们注意到，最先进的目标检测器通常对未知类进行分类，并且对可见类的可信度很高。尽管检测器是用一个背景类[54，13，32]显式训练的，和/或应用一个vs-rest分类器对每个类进行建模[14，30]。一个专门的工作机构[42，41，16]专注于开发目标检测器中（空间和语义）不确定性的度量，以拒绝未知类。例如，[42，41]在SSD检测器中使用蒙特卡罗差[11]采样来获得不确定度估计。然而，这些方法不能在一个动态的世界中逐渐调整它们的知识。**

## 3 开放世界目标检测

### 3-1 公式定义

Let us formalise the definition of Open World Object Detection in this section. At any time $t$, <font color="red">we consider the set of known object classes as $\mathcal{K}^t=\{1,2,..,\mathbb{C}\}\subset\mathbb{N}^+$ </font>where $\mathbb{N}^+$ denotes the set of positive integers. In order to realistically model the dynamics of real world, we also assume that their exists a set of <font color="red">unknown classes $\mathcal{U}=\{\mathbb{C}+1,...\}$</font>, which may be encountered during inference. The <font color="red">known object classes $K_t$ are assumed to be labeled in the dataset $\mathcal{D}^t\:=\:\{\mathbf{X}^t,\mathbf{Y}^t\}$ where $\mathbf{X}$ and $Y$ denote the input images and labels respectively. The input image set comprises of $M$ training images, $\mathbf{X}^t=\{I_1,\ldots,I_M\}$ </font>and associated object labels for each image forms the label set<font color="red"> $\mathbf{Y}^t=\{\mathbf{Y}_1,\ldots,\mathbf{Y}_M\}$.</font> Each $Y_i=\{\boldsymbol{y}_1,\boldsymbol{y}_2,..,\boldsymbol{y}_K\}$ encodes a set of $K$ object instances with their class labels and locations i.e., <font color="red">$y_k=[l_k,x_k,y_k,w_k,h_k]$,</font> where $l_k\in\mathcal{K}^t$ and $x_k,y_k,w_k,h_k$ denote the bounding box center coordinates, width and height respectively.

**在本节中正式定义开放世界目标检测。在任意时刻 $t$,<font color="red">我们将已知的目标类集合记为 $\mathcal{K}^t=\{1,2,..,\mathbb{C}\}\subset\mathbb{N}^+$</font> ,其中，$\mathbb{N}^+$表示正整数集合。为了真实地模拟现实世界的动态，我们还假设它们存在一组<font color="red">未知的类别$\mathcal{U}=\{\mathbb{C}+1,...\}$</font>,这在推理过程中可能会遇到。假设<font color="red">已知目标类$K_t$在数据集$D^t=\{X^t，Y^t\}$中被标记</font>,其中，<font color="red">$X$和$Y$分别表示输入图像和标签。输入图像集由$M$个训练图像组成，$X^t=\{I_1,...,I_M\}$</font>和每个图像的相关目标标签组成标签集$Y^t=\{Y_1,...,Y_M\}$。每个$Y_i=\{y_1,y_2,...,y_K\}$编码一组$K$目标实例及其类标签和未知，即<font color="red">$y_k=[l_k,x_k,y_k,w_k,h_k]$</font>,其中，$l_k\in\mathcal{K}^t$并且$x_k,y_k,w_k,h_k$分别表示边界框的中心坐标，宽度和高度。**

### 3-2 运行原理

The $\text{Open World Object Detection}$ setting considers an object detection model  $\mathcal{M}_{C}$ that is trained to detect all the previously encountered $C$ object classes. Importantly, <font color="red">the model $\mathcal{M}_{C}$ is able to identify a test instance belonging to any of the known  $C$ classes, and can also recognize a new or unseen class instance by classifying it as an $unknown$, denoted by a label zero (0). The unknown set of instances $\mathbf{U}^t$ can then be forwarded to a human user who can identify $n$ new classes of interest (among a potentially large number of unknowns) and provide their training examples. </font>The learner incrementally adds $n$ new classes and updates itself to produce an updated model $\mathcal{M}_{{C}+n}$ without retraining from scratch on the whole dataset. The known class set is also updated $K_{t+1}=K_t+\{{C}+1,\ldots,{C}+n\}.$ This cycle continues over the life of the object detector, where it adaptively updates itself with new knowledge. The problem setting is illustrated in the top row of Fig. 2.

**开放世界目标检测 设置考虑一个目标检测模型$\mathcal{M}_{C}$ ，它被训练来检测所有以前遇到的$C$个目标类别。重要的是，<font color="red">$\mathcal{M}_{C}$ 模型能够识别属于任何已知类集合$C$的测试实例，并且还可以通过将新的或者看不见的类实例分类为"未知"（用标签0表示）来识别它。然后，可以将未知实例$U^t$转发给人类用户，该用户可以识别$n$个新的感兴趣的类（可能有大量未知目标），并提供它们的训练示例。</font>学习器增量地添加$n$个新类并更新自己，以生成更新的模型$\mathcal{M}_{C+n}$,而无需从头开始对整个数据集进行再训练。已知的类集也会更新为 $K_{t+1}=K_t+\{{C}+1,\ldots,{C}+n\}$。这个循环会在目标检测器的整个生命周期中持续，在这个生命周期中，它会用新的知识自适应地更新自身。问题设置在图2的顶部表示**

##### 图2 方法概述

![image-20231030235937601](F:\论文\毕设学习\OWOD.assets\image-20231030235937601.png)

> **Top row**: At each incremental learning step, the model identifies unknown objects (denoted by ‘?’), which are progressively labelled (as blue circles) and added to the existing knowledge base (green circles). 
>
> **Bottom row**: Our open world object detection model identifies potential unknown objects using an energy-based classification head and the unknown-aware RPN. Further, we perform contrastive learning in the feature space to learn discriminative clusters and can flexibly add new classes in a continual manner without forgetting the previous classes.
>
> **上面一行：在每个增量学习步骤中，模型识别未知目标（用‘?’），逐步标记（蓝色圆圈）并添加到现有知识库（绿色圆圈）。**
>
> **下面一行：我们的开放世界目标检测模型使用基于能量的分类头和未知-感知的RPN来识别潜在的未知目标。此外，我们在特征空间中进行对比学习来判别类，并且可以灵活地连续添加新的类而不会忘记以前的类。**

## 4 ORE：开放世界目标检测器

### 4-0 概述

#### 4-0-1 两个挑战

A successful approach for Open World Object Detection should be able to identify unknown instances without explicit supervision and defy forgetting of earlier instances when labels of these identified novel instances are presented to the model for knowledge upgradation (without retraining from scratch). We propose a solution, ORE which addresses both these challenges in a unified manner.
**一个成功的开放世界目标检测方法应该能够在没有明确监督的情况下识别未知实例，并且在将这些识别出的新实例的标签提交给模型进行知识升级（无需从头开始再培训）时，能够克服对早期实例的遗忘。我们提出了一个解决方案，以统一的方式解决这两个挑战。**

#### 4-0-2 怎么应对——对比聚类

Neural networks are universal function approximators [21], which learn a mapping between an input and the output through a series of hidden layers. The latent representation learned in these hidden layers directly controls how each function is realised. We hypothesise that learning clear discrimination between classes in the latent space of object detectors could have two fold effect. First, <font color="red">it helps the model to identify how the feature representation of an unknown instance is different from the other known instances, which helps identify an unknown instance as a novelty. </font>Second, it <font color="red">facilitates learning feature representations for the new class instances without overlapping with the previous classes in the latent space, which helps towards incrementally learning without forgetting. </font>The key component that helps us realise this is our proposed <font color="red">contrastive clustering</font> in the latent space, which we elaborate in Sec. 4.1.

**神经网络是通过函数逼近器[21]，它通过一系列隐藏层学习输入和输出之间的映射。在这些隐藏层中学习的潜在表示直接控制每个功能的实现方式。我们假设，在目标检测器的潜在空间中学习类间的清晰区分可能产生双重效果。首先，它<font color="red">帮助模型识别未知实例的特征表示与其他已知实例的区别，从而有助于将未知实例识别为新实例。</font>第二，它<font color="red">有助于学习新类实例的特征表示，而不与潜在空间中的前一类重叠，从而有助于不遗忘的增量学习。</font>帮助我们认识到这一点的关键部分是我们在潜在空间中提出的<font color="red">对比聚类</font>，我们将在第二节中详细阐述**

#### 4-0-3 怎么应对——使用RPN自动标记未知

Fig. 2 shows the high-level architectural overview of ORE. We <font color="red">choose Faster R-CNN [53] as the base detector</font> as Dhamija et al. [7] has found that it has better open set performance when compared against one-stage RetinaNet detector [30] and objectness based YOLO detector [51]. <font color="red">Faster R-CNN [53] is a two stage object detector. In the first stage, a class-agnostic Region Proposal Network (RPN) proposes potential regions which might have an object from the feature maps coming from a shared backbone network. The second stage classifies and adjusts the bounding box coordinates of each of the proposed region. The features that are generated by the residual block in the Region of Interest (RoI) head are contrastively clustered. The RPN and the classification head is adapted to auto-label and identify unknowns respectively.</font> We explain each of these coherent constituent components, in the following subsections:
**图2显示了ORE的高级架构概述。我们<font color="red">选择Faster R-CNN[53]作为基本检测器</font>，因为Dhamija等人[7]发现，与单级RetinaNet检测器[30]和基于对象的YOLO检测器[51]相比，它具有更好的开集性能。<font color="red">Faster R-CNN[53]是一个两阶段目标检测器。在第一阶段中，类-无关区域建议网络（RPN）提出可能具有来自共享主干网络的特征映射的目标的潜在区域。第二阶段对每个区域的边界跨国坐标进行分类和调整。对感兴趣区域（Rol）模块其他部分生成的特征进行对比聚类。RPN和分类头分别用于自动标注和识别未知量</font>。我们将在以下小节中解释这些连贯的组成部分：**

### 4-1 Contrastive Clustering对比聚类

Class separation in the latent space would be an ideal characteristic for an Open World methodology to identify unknowns. A natural way to enforce this would be to model it as a contrastive clustering problem, where instances of same class would be forced to remain close-by, while instances of dissimilar class would be pushed far apart.
**潜在空间中的类分离是开放世界方法识别未知的理想特征。一种自然的方法是将其建模为一个对比聚类问题，在这个问题中，同一类的实例将被迫保持在附近，而不同类的实例将被推得很远。**

  For each known class $i\in\mathcal{K}^t$, we maintain a prototype vector $p_i$. Let $f_c\in\mathbb{R}^d$ be a feature vector that is generated by an intermediate layer of the object detector, for an object of class $c$. We define the contrastive loss as follows:

**对于每个已知类$i\in\mathcal{K}^t$，我们有一个原型向量$p_i$。令$f_c\in\mathbb{R}^d$为特征检测器的中间层为$c$类物体生成的特征向量。我们将对比损失定义如下：**

$\begin{gathered}
\mathcal{L}_{cont}(\boldsymbol{f}_{c}) =\sum_{i=0}^{\mathsf{C}}\ell(f_{c},p_{i}),\mathrm{~where}, \text{(1)} \\
\ell(\boldsymbol{f_{c}},\boldsymbol{p_{i}}) =\begin{cases}\mathcal{D}(\boldsymbol{f}_c,\boldsymbol{p}_i)&i=c\\\max\{0,\Delta-\mathcal{D}(\boldsymbol{f}_c,\boldsymbol{p}_i)\}&\mathrm{otherwise}\end{cases} 
\end{gathered}$

defines how close where $\mathcal{D}$ is any distance function and $\Delta$ defines how close a similar and dissimilar item can be. Minimizing this loss would ensure the desired class separation in the latent space.

**其中$\mathcal{D}$ 是任何距离函数，$\Delta$定义了相似和不相似项之间的距离。最小化这种损失将确保在潜在空间中实现所需的类分离**

<font color="red">Mean of feature vectors corresponding to each class is used to create the set of class prototypes: $\mathcal{P} = \{ p_0\cdots p_\mathrm{C} \} .$</font> Maintaining each prototype vector is a crucial component of ORE. <font color="red">As the whole network is trained end-to-end, the class prototypes should also gradually evolve, as the constituent features change gradually</font> (as stochastic gradient descent updates weights by a small step in each iteration). We maintain a<font color="red"> fixed-length queue $q_i$, per class for storing the corresponding features. A feature store $\mathcal{F}_{store} =\{\boldsymbol{q}_0\cdots\boldsymbol{q}_\mathrm{C}\}$, stores the class specific features in the corresponding queues</font>. This is a scalable approach for keeping track of how the feature vectors evolve with training, as the number of feature vectors that are stored is bounded by $C\times Q$, where $Q$ is the maximum size of the queue.

**<font color="red">每个类对应的特征向量的平均值被用来创建类原型集：$\mathcal{P} = \{ p_0\cdots p_\mathrm{C} \} $</font>。生成的每个原型向量是ORE的一个关键组成部分。<font color="red">随着整个网络的端到端训练，类原型也应该随着组成特征的逐渐变化而逐渐演化</font>（因为随机梯度下降在每次迭代中更新一小步权重）。我们有一个<font color="red">固定长度的队列 $q_i$,每个类用于存储相应的特征。特征集$\mathcal{F}_{store} =\{\boldsymbol{q}_0\cdots\boldsymbol{q}_\mathrm{C}\}$将类特定的特性存储在相应的队列中。</font>这是一种可伸缩的方法，用于跟踪特征向量如何随训练而演化，因为存储的特征向量的数量以$C\times Q$为界，其中$Q$是队列的最大尺寸。**

Algorithm 1 provides an overview on how class prototypes are managed while computing the clustering loss.<font color="red"> We start computing the loss only after a certain number of burnin iterations $(I_b)$ are completed.</font> This allows the initial feature embeddings to mature themselves to encode class information. Since then, we compute the clustering loss using Eqn. l. After every $I_p$ iterations, a set of new class prototypes $P_{new}$ is computed (line 8). Then the existing prototypes $\mathcal{P}$ are updated by weighing $\mathcal{P}$ and $\mathcal{P}_{new}$ with a momentum parameter $\eta$. This allows the class prototypes to evolve gradually keeping track of previous context. The computed clustering loss is added to the standard detection loss and back-propagated to learn the network end-to-end.

##### 算法1

**算法1 概述了在计算集群损失时如何管理类原型。只有<font color="red">在完成一定数量的burnin迭代($I_b$)之后，我们才开始计算损耗</font>。这使得初始的特征映射能够逐渐准确，从而对类信息进行编码。从那时起，我们使用公式1计算聚类损失。在每个$I_p$迭代之后，计算一组新的类原型$\mathcal{P}_{new}$（第8行）。然后用动量参数 $\eta$ 对 $\mathcal{P}$和$\mathcal{P}_{new}$进行加权，更新现有的原型$ \mathcal{P} $ 。这允许类原型逐渐演化，并跟踪以前的上下文信息。将计算出的聚类损失加入到标准检测损失中，并进行反向传播，实现对网络的端到端学习。**

------

**Algorithm 1** Algorithm COMPUTEC LUSTERING LOSS 

------

**Input**: Input feature for which loss is computed: $f_c;$ Feature
 store: $\mathcal{F}_{store};$ Current iteration: $i;$ Class prototypes: $\mathcal{P}=$ $\{\boldsymbol{p}_0\cdots\boldsymbol{p}_\mathbb{C}\};$ Momentum parameter: $\eta.$

1: Initialise $\mathcal{P}$ if it is the first iteration. 
2: $\mathcal{L}_{cont}\gets0$
 3: **if** $i==I_b$ **then** 
4:$\qquad\mathcal{P}\leftarrow\mathsf{class-wise~mean~of~items~in~}\mathcal{F}_{Store}. .$
 5: $\qquad\mathcal{L}_{cont}\gets\mathsf{Compute}$ using $f_c,\mathcal{P}$ and Eqn. 1. 
6: **else if** $i>I_b$ then
 7: $\qquad$**if** $i\%I_p==0$ **then**
 8: $\qquad\qquad\mathcal{P}_{new}\gets$ class-wise mean of items in $\mathcal{F}_{Store}.$
 9: $\qquad\qquad\mathcal{P}\gets\eta\mathcal{P}+(1-\eta)\mathcal{P}_{new}$
10:$\qquad\mathcal{L}_{cont}\leftarrow\mathsf{Compute\,using\,}\boldsymbol{f}_c,\mathcal{P}\,\mathrm{and}\,\mathsf{Eqn.~1.}$
 11: **return** $\mathcal{L}_{cont}$

-------

### 4-2 Auto-labelling Unknowns with RPN使用RPN自动标记未知

While computing the clustering loss with Eqn. 1, we contrast the input feature vector $f_c$ against prototype vectors, which include a prototype for unknown objects too $(c\in\{0,1,..,{C}\}$ where 0 refers to the unknown class). This would require unknown object instances to be labelled with unknown ground truth class, which is not practically feasible owing to the arduous task of re-annotating all instances of each image in already annotated large-scale datasets.

**在计算聚类损失时，用公式1，我们<font color="red">将输入特征向量$f_c$与原型向量进行对比</font>，原型向量也包括未知目标的原型 （$c\in\{0,1,..,{C}\}$，其中0表示未知类）。这将要求未知目标实例被标记为未知标准类，这实际上是不可行的，因为在已经注释的大规模数据集中重新注释每个图像的所有实例困难。 **

As a surrogate, we propose to automatically label some of the objects in the image as a potential unknown object. For this, we rely on the fact that Region Proposal Network (RPN) is class agnostic. Given an input image, the RPN generates a set of bounding box predictions for foreground and background instances, along with the corresponding objectness scores. We label those proposals that have high objectness score, but do not overlap with a ground-truth object as a potential unknown object. Simply put, we select the top-k background region proposals, sorted by its objectness scores, as unknown objects. This seemingly simple heuristic achieves good performance as demonstrated in Sec. 5.
**作为代理，我们建议自动将图像中的一些目标标记为潜在的未知目标。为此，我们使用区域建议网络（RPN）因为它与类无关。<font color="red">给定一个输入图像，RPN为前景和背景实例生成一组边界框预测，以及相应的目标得分。我们将那些具有高目标性得分，但不与ground truth重叠的区域标记为潜在未知目标。简单地说，我们选择top-k背景区域方案，按其目标得分排序，作为未知对象</font>。就如第五部分所讲，这个看似简单的启发式方法可以获得很好的性能**

### 4-3  Energy Based Unknown Identifier 基于能量的未知目标识别

Given the features $(\boldsymbol{f}\in F)$ in the latent space $F$ and their corresponding labels $l\in L$, we seek to learn an energy function $E(F,L).$ Our formulation is based on the Energy based models (EBMs) [27] that learn a function $E(\cdot)$ to estimates the compatibility between observed variables $F$ and possible set of output variables $L$ using a single output scalar i.e., $E(\boldsymbol{f}):\mathbb{R}^d\to\mathbb{R}.$ The intrinsic capability of EBMs to assign low energy values to in-distribution data and vice-versa motivates us to use an energy measure to characterize whether a sample is from an unknown class.

Specifically, we use the Helmholtz free energy formulation where energies for all values in $L$ are combined,

**给定隐空间$F$的特征$(\boldsymbol{f}\in F)$及其相应的标号$l\in L$，我们寻求学习一个能量函数$E(F,L)$。我们的公式<font color="red">基于能量的模型（EBMs）[26]</font>，该模型学习一个函数$E(\cdot)$，以使用单个输出标量估计观测变量$F$和可能的输出变量集$L$之间的兼容性，即$E(\boldsymbol{f}):\mathbb{R}^d\to\mathbb{R}$。EBMs向分布内数据分配低能量值的内在能力，反之亦然，促使我们使用能量度量来表征样本是否来自未知类别。 **

**具体来说，我们使用亥姆霍兹自由能公式，其中$L$中所有值的能量都是组合的**

$E(\boldsymbol{f})=-T\log\int_{l'}\exp\left(-\frac{E(\boldsymbol{f},l')}{T}\right),\quad(2)$

where T is the temperature parameter. There exists a simple relation between the network outputs after the softmax layer and the Gibbs distribution of class specific energy values [33]. This can be formulated as
**其中T是温度参数。softmax层之后的网络输出与类比能量值的Gibbs分布之间存在简单的关系[33]。这可以表述为**

$p(l|\boldsymbol{f})=\frac{\exp(\frac{g_l(\boldsymbol{f})}T)}{\sum_{i=1}^\mathbf{C}\exp(\frac{g_i(\boldsymbol{f})}T)}=\frac{\exp(-\frac{E(\boldsymbol{f},l)}T)}{\exp(-\frac{E(\boldsymbol{f})}T)}\quad\mathrm{~(3)}$

where $p(l|\boldsymbol{f})$ is the probability density for a label $l,g_l(\boldsymbol{f})$ is the $l^{th}$ classification logit of the classification head $g(.).$ Using this correspondence, we define free enercy of our  classification models in terms of their logits as follows:

**其中$p(l|\boldsymbol{f})$是标签$l$的概率密度，$g_l(\boldsymbol{f})$是分类头$g(.)$的第$l$次分类回归。利用这种对应关系，我们用logit定义分类模型的自由能，如下所示：**

$E(\boldsymbol{f};g)=-T\log\sum_{i=1}^\mathrm{C}\exp(\frac{g_i(\boldsymbol{f})}T).\quad\quad(4)$

The above equation provides us a natural way to transform the classification head of the standard Faster R-CNN [54] to an energy function. Due to the clear separation that we enforce in the latent space with the contrastive clustering, we see a clear separation in the energy level of the known class data-points and unknown data-points as illustrated in Fig. 3. In light of this trend, we <font color="red">model the energy distribution of the known and unknown energy values $\xi_{kn}(\boldsymbol{f})$ and $\xi_{unk}(\boldsymbol{f})$, with a set of shifted Weibull distributions. </font>These distributions were found to fit the energy data of a small held out validation set (with both knowns and unknowns instances) very well, when compared to Gamma, Exponential and Normal distributions. <font color="red">The learned distributions can be used to label a prediction as unknown if $\xi_{kn}(\boldsymbol{f})<\xi_{unk}(\boldsymbol{f}).$</font>

**上面的公式为我们提供了一种自然的方法，将Faster R-CNN[53]的分类头转换为能量函数。由于我们使用对比聚类在潜在空间中实施的清晰分离，我们看到如图3所示的已知类数据点和未知数据点的能级中的清晰分离。根据这一趋势，我们<font color="red">用一组移位的Weibull分布来模拟已知和未知能量值$\xi_{kn}(\boldsymbol{f})$和$\xi_{unk}(\boldsymbol{f})$的能量分布。</font>与伽马分布、指数分布和正态分布相比，这些分布与验证集的能量数据非常吻合。<font color="red">如果$\xi_{kn}(\boldsymbol{f})<\xi_{unk}(\boldsymbol{f})$，则学习的分布可用于将预测标记为未知。</font>**

##### 图3 

![image-20231101001717159](F:\论文\毕设学习\OWOD.assets\image-20231101001717159.png)

> Figure 3: The energy values of the known and unknown datapoints exhibit clear separation as seen above. We fit a Weibull distribution on each of them and use these for identifying unseen known and unknown samples, as explained in Sec. 4.3
> **图3：如上图所示，已知和未知数据点的能量值显示出明显的分离。我们在每个样本上拟合一个Weibull分布，并用这些来识别未知样本和已知样本，如第4.3节所述**

