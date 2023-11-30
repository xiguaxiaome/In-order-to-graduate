[TOC]



# OW-DETR：Open-world DetectionTransformer

Transformer的开放世界目标检测

# 0、概述Abstract

Open-world object detection (OWOD) is a challenging computer vision problem, where the task is to detect a known set of object categories while simultaneously identifying unknown objects.

开放世界目标检测(**O**pen-**w**orld **o**bject **d**etection：**OWOD**)是一个具有挑战性的计算机视觉问题，其任务是检测一组已知的对象类别，同时识别未知对象。

 Additionally, the model must incrementally learn new classes that become known in the next training episodes. 

此外，模型必须增量地学习在下一个训练集中变得已知的新类。

Distinct from standard object detection, the OWOD setting poses significant challenges for generating quality candidate proposals on potentially unknown objects, separating the unknown objects from the background and detecting diverse unknown objects. 

与标准目标检测目标不同，OWOD提出了重要的挑战：**对潜在的未知对象生成高质量候选框建议**，以及**对潜在的未知对象从背景中分离出来**以及**检测不同的未知目标**。

Here, we introduce a novel end-to-end transformer-based framework, OW-DETR, for open-world object detection. 

在这里，我们介绍一个新的**端到端的基于Transformer的框架**，叫OW-DETR，用于开放世界目标检测。

The proposed OW-DETR comprises three dedicated components namely, attention-driven pseudo-labeling, novelty classification and objectness scoring to explicitly address the aforementioned OWOD challenges.

本文提出的OW-DETR有三个专门的部分组成：

**:one:注意力驱动的伪标注(attention-driven pseudo-labeling)**、**:two:新类别分类(novelty classification)**、**:three:目标性评分(objectness scoring)**，以明确解决上述OWOD挑战。

 Our OW-DETR explicitly encodes multi-scale contextual information, possesses less inductive bias, enables knowledge transfer from known classes to the unknown class and can better discriminate between unknown objects and background. 

我们的OW-DETR显示编码了**多尺度上下文信息(multi-scale contextual infomation)**,具有较少的**归纳偏置(inductive bias)**，可以实现从已知类到未知类的知识转移，并且可以更好地区分未知目标和背景。

Comprehensive experiments are performed on two benchmarks: MS-COCO and PASCAL VOC. 

我们在两个基准数据集上做了综合实验：MS-COCO和PASCAL VOC。

The extensive ablations reveal the merits of our proposed contributions.

广泛的消融实验显示了我们提出的的贡献的优势。

 Further, our model outperforms the recently introduced OWOD approach, ORE, with absolute gains ranging from 1.8% to 3.3% in terms of unknown recall on MS-COCO.

此外，我们的模型优于最近的OWOD模型：ORE，在MS-COCO数据集上的未知类别召回率(unknown recall)方面的绝对增益子啊1.8%到3.3%之间。

 In the case of incremental object detection, OW-DETR outperforms the state-of-the-art for all settings on PASCAL VOC. 

在增量目标检测的情况下，对于PASCAL VOC的所有设置，OW-DETR的性能都处于最优的情况。

Our code is available at https://github.com/akshitac8/OW-DETR.

略。

# 1、介绍Introduction

### 图1 OW-DETR的可视化说明

![image-20231127224527051](.\\OW-DETR：Open-world DetectionTransformer.assets\image-20231127224527051.png)

Figure 1. **Visual illustration of the proposed OW-DETR for open-world object detection (OWOD)**. 

图1. **用于开放世界目标检测(OWOD)的OW-DETR可视化说明**。

Here, attention maps obtained from the intermediate features are utilized to score the object queries.

在这里，利用从中间特征(intermediate features)获得的注意图(attention maps)来对目标查询进行评分。

 The objectness scores of queries are then used to identify the pseudo-unknowns. 

然后使用查询的目标分数来识别伪未知目标。

A separation is enforced between these pseudo-unknowns and ground-truth knowns to detect novel classes. 

在这些伪未知目标和已知目标之间强制分离，以检测新的类。

In addition, a separation is also learned between the background and foreground (knowns + unknowns) for effective knowledge transfer from known to unknown class w.r.t. characteristics of foreground objects.

此外，还学习了背景和前景（已知+未知）的分离，以便有效地将前景对象的已知类w.r.t.特征知识转移到未知类。

 Our OW-DETR explicitly encodes multiscale context, has less inductive bias, and assumes no supervision for unknown objects, thus well suited for OWOD problem.

我们的OW-DETR显示地对多尺度上下文进行编码，具有较小的归纳偏置，并且不需要对未知对象进行监督，因此非常适合于OWOD问题。

## 1.1 第一段 开放世界目标检测OWOD介绍

Open-world object detection (OWOD) relaxes the closed-world assumption in popular benchmarks, where only seen classes appear at inference.

开放世界目标检测(**O**pen-**w**orld **o**bject **d**etection)放宽了流行的基准测试中的封闭世界假设，在封闭世界假设中只有看到（被标注类别）的类出现在推断中。

> 也就是我标了啥物体模型就能学会啥物体，没标的物体都识别为背景。

Within the OWOD paradigm [15], at each training episode, a model learns to detect a given set of known objects while simultaneously capable of identifying unknown objects.

在OWOD的范例中，在每个训练集中，模型学习检测给定的一组已知对象，同时能够识别未知对象。

 These flagged unknowns can then be forwarded to an oracle (e.g., human annotator), which can label a few classes of interest.

这些标记的未知类可以被转发给oracle（人工标注器），它可以标记一些感兴趣的类。

 Given these new knowns, the model would continue updating its knowledge incrementally without retraining from scratch on the previously known classes. 

给定这些新的已知类，模型将继续增量地更新其知识，而无需在先前已知的类上从头开始重新训练。

This iterative learning process continues in a cycle over the model’s life-span.

这个迭代的学习过程在模型的生命周期内循环进行。

## 1.2 第二段  OWOD挑战(3条)

The identification of unknown object classes in OWOD setting poses significant challenges for conventional detectors. 

OWOD下的未知目标类别的识别对传统检测器提出了重大挑战。

First, besides an accurate proposal set for seen objects, a detector must also generate quality candidate boxes for potentially unknown objects. 

首先，除了为已看到的物体提供准确的建议框集合外，检测器还必须为潜在未知的目标生成高质量的候选框。

Second, the model should be able to separate unknown objects from the background utilizing its knowledge about the already seen objects, thereby learning what constitutes a valid object.

第二，该模型将未知对象从背景中分离出来，通过利用其关于已看到的对象的知识，从而了解构成有效对象的内容。

Finally, objects of different sizes must be detected while flexibly modeling their rich context and relations with co-occurring objects.

最后，必须检测不同大小的对象，同时灵活地建模其丰富的上下文和与共发生的对象的关系。

## 1.3 第三段 OWOD鼻祖(ORE)介绍

Recently, the work of [15] introduces an open-world object detector, ORE, based on the two-stage Faster RCNN [32] pipeline. 

最近，《Towards open world object detection》的工作介绍了一种基于 两阶段Faster RCNN 管道的开放世界目标检测器ORE。 

Since unknown object annotations are not available during training in the open-world paradigm, ORE proposes to utilize an auto-labeling step to obtain a set of pseudo-unknowns for training. 

由于在开放世界范式的训练过程中，未知目标标注是不可用的，因此ORE建议利用自动标记(auto-labeling)步骤来获得一组用于训练的伪未知目标(pseudo-unknowns)。

> 伪未知目标(pseudo-unknowns)：就是目标检测器认为虽然不在已知类别里，但是这个东西长得还挺像个东西的，确实不是个背景。

The auto-labeling is performed on class-agnostic proposals output by a region proposal network (RPN). 

自动标注是由区域建议网络(RPN)输出的类别无关的建议框们来执行的。

The proposals not overlapping with the ground-truth (GT) known objects but having high ‘objectness’ scores are auto-labeled as unknowns and used in training.

不与真实(GT)已知物体重叠但具有高"目标性"分数的建议被自动标记为“未知”并用于训练。

 These auto-labeled unknowns are then utilized along with GT knowns to perform latent space clustering.

然后这些自动标记的未知物体与已知的真实物体(GT)一起用于执行潜在的空间聚类。

Such a clustering attempts to separate the multiple known classes and the unknown class in the latent space and aids in learning a prototype for the unknown class. 

这种聚类试图在潜在空间中分离多个已知类和未知类，并帮助学习未知类的原型。

Furthermore, ORE learns an energy-based binary classifier to distinguish the unknown class from the class-agnostic known class.

此外，ORE学习了一个基于能量的二进制分类器来区分来自 类无关的已知类中的未知类。

## 1.4 第四段 ORE的问题(3条)

While being the first to introduce and explore the challenging OWOD problem formulation, ORE suffers from several issues. 

虽然ORE是第一个引入和探索具有挑战性的OWOD问题表述的，但它存在几个问题：

(i) ORE relies on a held-out validation set with weak supervision for the unknowns to estimate the distribution of novel category in its energy-based classifier.

(i)ORE依赖一个带有对未知目标弱监督的保留验证集来估计新类别的分布，基于能量分类器。

 (ii) To perform contrastive clustering, ORE learns the unknown category with a single latent prototype, which is insufficient to model the diverse intra-class variations commonly present in the unknown objects. 

 (ii) 为了对比聚类，ORE使用单个潜在原型学习未知目标的类别，这不足以对未知对象中常见的多种类变化进行建模。

> 大概是把所有的未知目标都看成一类了，感觉不太符合实际。

Consequently, this can lead to a sub-optimal separation between the knowns and unknowns.

因此，这可能导致已知和未知之间的次优分离。

> 什么是次优分离？？
>
> 我猜大概是已知目标中每个类分的很清楚猫是猫狗是狗的，但是未知类是一坨

 (iii) ORE does not explicitly encode long-range dependencies due to a convolution-based design, crucial to capture the contextual information in an image comprising diverse objects. 

(iii)由于基于卷积的设计，ORE没有明确编码远程依赖关系，这对于捕获包含不同对象的图像中的上下文信息至关重要。

Here, we set out to alleviate the above issues for the challenging OWOD problem formulation.

在这里，我们着手缓解上述问题，以解决具有挑战性的OWOD问题表述。

## 1.5 第五段 我们的OW-DETR的贡献(3条)

**Contributions**: Motivated by the aforementioned observations, we introduce a multi-scale context aware detection framework, based on vision transformers [38], with dedicated components to address open-world setting including attention-driven pseudo-labeling, novelty classification and objectness scoring for effectively detecting unknown objects in images (see Fig. 1).

**贡献**：受上述观察结果的启发，我们引入了一种基于视觉Transformer的多尺度上下文感知检测框架,该框架具有解决开放世界设置的专用组件，包括**注意力驱动的伪标注**、**新类别分类**和**目标性评分**，以有效检测图像中的未知目标(见图1)。

 Specifically, in comparison to the recent OWOD approach ORE [15], that uses a two-stage CNN pipeline, ours is a single-stage framework based on transformers that require less inductive biases and can encode long-term dependencies at multi-scales to enrich contextual information. 

具体而言，与最近使用两阶段CNN管道(two-stage CNN pipeline)的OWOD方法ORE相比，我们的是一个基于Transformer的单阶段框架，它需要较少的归纳偏置，并且可以在多尺度上对长期依赖关系进行编码，以丰富上下文信息。

Different to ORE, which relies on a held-out validation set for estimating the distribution of novel categories, our setting assumes no supervision given for the unknown and is closer to the true open-world scenario. 

与ORE不同的是，ORE依赖于一个保留的验证集来估计新类别的分布，我们的设置假设对未知目标不监督，因此更接近真实的开放世界场景。

Overall, our novel design offers more flexibility with broad context modeling and less assumptions to address the open-world detection problem.Our main contributions are:

总的来说，我们的新设计为解决开放世界检测问题提供了更大的灵活性和更少的假设。我们的主要贡献有：

• We propose a transformer-based open-world detector, OW-DETR, that better models the context with mutliscale self-attention and deformable receptive fields, in addition to fewer assumptions about the open-world setup along with reduced inductive biases.

• 我们提出了一种基于Transformer的开放世界目标检测器OW-DETR，它可以更好地模拟具有多尺度自注意和可变形的接受域的环境，此外还可以减少对开放世界设置的假设以减少归纳偏置。

• We introduce an attention-driven pseudo-labeling scheme for selecting the object query boxes having high attention scores but not matching any known class box as unknown class. 

• 我们引入了一个注意力驱动的伪标注方案，用于选择具有高注意力分数但不属于任何已知类目标框的目标查询(object query)匹配为未知类。

The pseudo-unknowns along with the ground-truth knowns are utilized to learn a novelty classifier to distinguish the unknown objects from the known ones.

利用伪未知类和真的(GT)已知类来学习新类别分类器来区分未知目标和已知目标。

• We introduce an objectness branch to effectively learn a separation between foreground objects (knowns, pseudo-unknowns) and the background by enabling knowledge transfer from known classes to the unknown class w.r.t. the characteristics that constitute a foreground object.

•我们引入了一个目标性分支，通过使知识从已知类转移到未知类（构成前景目标的特征），从而有效地学习前景目标(已知的，伪未知的)和背景之间的分离。

• Our extensive experiments on two popular benchmarks demonstrate the effectiveness of the proposed OWDETR. 

•我们在两个流行的基准数据集上进行了广泛的实验，证明了所提出的OWDETR的有效性。

Specifically, OW-DETR outperforms the recently introduced ORE for both OWOD and incremental object detection tasks.

具体来说，对于OWOD和增量目标检测任务，OW-DETR优于最近引入的ORE。

 On MS-COCO, OW-DETR achieves absolute gains ranging from 1.8% to 3.3% in terms of unknown recall over ORE.

在MS-COCO上，与ORE相比，ow - detr在未知类召回率方面实现了1.8%至3.3%的绝对增益。

# 2、基于Transformer的开放世界目标检测Open-world Detection Transformer

## 2.0 概述

### 2.0.1 第一段 问题描述

**Problem Formulation**: Let $K^t=\{1,2,\cdots,C\}$ denote the set of known object categories at time $t$ . 

**问题描述**：设$K^t=\{1,2,\cdots,C\}$ 表示时间 $t$ 时已知的目标类别集合。

Let $\mathcal{D}^t=\{\mathcal{I}^t,\mathcal{Y}^t\}$ be a dataset containing $N$ images $\mathcal{I}^t=\{I_1,\cdots,I_N\}$ with corresponding labels $\mathcal{Y}^t=\{Y_1,\cdots,Y_N\}.$ 

设 $\mathcal{D}^t=\{\mathcal{I}^t,\mathcal{Y}^t\}$ 是一个包含$N$ 张图像$\mathcal{I}^t=\{I_1,\cdots,I_N\}$ 及其对应标签$\mathcal{Y}^t=\{Y_1,\cdots,Y_N\}$的数据集。

Here, each $Y_i=\{y_1,\cdots,y_K\}$ denotes the labels of a set of $K$ object instances annotated in the image with $y_k=$ $[l_k,x_k,y_k,w_k,h_k]$, where $l_k\in\mathcal{K}^t$ is the class label for a bounding box represented by $x_k,y_k,w_k,h_k.$ 

其中，每个 $Y_i=\{y_1,\cdots,y_K\}$ 表示图像中标注的一组 $K$ 个对象实例的标签， $y_k=$ $[l_k,x_k,y_k,w_k,h_k]$, 其中 $l_k\in\mathcal{K}^t$ 是由$x_k,y_k,w_k,h_k$表示的边界框的类别标签。

Furthermore, $\det\mathcal{U}=\{C+1,\cdots\}$ denote a set of unknown classes that might be encountered at test time.

此外， 定义$\mathcal{U}=\{C+1,\cdots\}$ 表示可能在测试时遇到的未知类别集合。

### 2.0.2 第二段 开放世界检测流程

As discussed in Sec. 1, in the open-world object detection (OWOD) setting, a model $M^t$ at time $t$ is trained to identify an unseen class instance as belonging to the unknown class (denoted by label 0), in addition to detecting the previously encountered known classes $C.$ 

正如在第一节中讨论的，在开放世界目标检测(OWOD)设置下，除了检测以前遇到的已知类别$C$类外。还训练$t$时刻的模型$M^t$来识别一个没见过的类别实例，该实例属于未知类(用标签0表示)。

A set of unknown instances $U^t\subset\mathcal{U}$ identified by $M^t$ are then forwarded to an oracle, which labels $n$ novel classes of interest and provides a corresponding set of new training examples. 

模型$M^t$识别出的一组未知实例 $U^t\subset\mathcal{U}$ 被发送给一个oracle，该oracle（人工标注器）标注了$n$个感兴趣的新类别，并提供相应的新训练样本集。

The learner then incrementally adds this set of new classes to the known classes such that $K^{t+1}=K^t+\{C+$ $1,\cdots,C+n\}.$ 

这个学习者随后将这组新类别逐步添加到已知类别中，使得$K^{t+1}=K^t+\{C+$ $1,\cdots,C+n\}$。

For the previous classes $K^t$, only few examples can be stored in a bounded memory, mimicking privacy concerns, limited compute and memory resources in realworld settings. 

对于之前的类别 $K^t$，只能在有限的内存中存储少量示例，以模拟真实世界中的隐私问题、计算和内存资源的限制。

Then, $\mathcal{M}^t$ is incrementally trained, without retraining from scratch on the whole dataset, to obtain an updated model $\mathcal{M}^{t+1}$ which can detect all object classes in $\mathcal{K}^{t+1}$.

然后， $\mathcal{M}^t$进行增量训练，而不是从头开始对整个数据集重新训练，以获得更新的模型 $\mathcal{M}^{t+1}$ ,该模型可以检测到 $\mathcal{K}^{t+1}$中所有目标类别。

 This cycle continues over the life-span of the detector, which updates itself with new knowledge at every episode without forgetting the previously learned classes.

这个循环在目标检测器的整个生命中期中持续进行，它在每一集都用新知识更新自己，不会忘记以前学过的课程。

#### 图2 OW-DETR框架

![image-20231128153551874](.\\OW-DETR：Open-world DetectionTransformer.assets\image-20231128153551874.png)



Figure 2. **Proposed OW-DETR framework.** Our approach adapts the standard Deformable DETR for the OWOD problem formulation by introducing

图2。**提出的OW-DETR框架**。我们的方法通过将标准的可变性DETR(DDETR)用于开放世界目标检测问题(OWOD)：

 (i) an attention driven pseudo-labeling scheme to select the candidate unknown queries,

 (i) 基于注意力的伪标注方案，用于选择候选的未知目标查询；

 (ii) a novelty classification branch $F_{cls}$ to distinguish the pseudo unknowns from each of the known classes and 

 (ii) 新类别分类分支 $F_{cls}$ ，用于区分伪未知类别和每个已知类别；

(iii)  $F_{obj}$ that learns to separate foreground objects (known + pseudo unknowns) from the background.

(iii) 目标性分支 $F_{obj}$ ，学习将前景目标（已知类别和伪未知类别）与背景分离。

 In our OW-DETR, $D$-dimensional multi-scale features for an image $I$ are extracted from the backbone and input to the deformable encoder-decoder along with a set of $M$ learnable object queries $q\in\mathbb{R}^D$ to the decoder. 

在我们的 OW-DETR中,从主干网络中提取图像$I$的 $D$-维多尺度特征，并将其与一组$M$个可学习的对象查询  $q\in\mathbb{R}^D$ 一起输入到可变形编码器-解码器中。

At the decoder output, each object query embedding $q_e\in\mathbb{R}^D$ is input to three different branches: box regression, novelty classification and objectness. 

在解码器输出时，每个对象查询嵌入 $q_e\in\mathbb{R}^D$ 被输入到三个不同的分支：边界框回归、新颖性分类和目标性。

The box co-ordinates are output by the regression branch $F_{reg}$. 

边界框回归分支 $F_{reg}$输出边界框坐标。 

The objectness branch outputs the confidence of a query being a foreground object, whereas the novelty classification branch classifies the query into one of the known and unknown classes. 

目标性分支输出一个查询作为前景对象的置信度，而新颖性分类分支将查询分类为已知类别或未知类别之一。

Our OW-DETR is jointly learned end-to-end with novelty classification loss $\mathcal{L}_n$, objectness loss $\mathcal{L}_o$ and box regression loss $\mathcal{L}_r.$

我们的 OW-DETR 通过与新颖性分类损失 $\mathcal{L}_n$, 目标性损失 $\mathcal{L}_o$ 和边界框回归损失 $\mathcal{L}_r$一起进行端到端的联合学习。

## 2.1 总体架构

### 2.1.1 第一段

Fig. 2 shows the overall architecture of the proposed open-world detection transformer, OW-DETR. 

图2显示了提出的基于Transformer的开放世界检测器，OW-DETR的总体架构。

The proposed OW-DETR adapts the standard Deformable DETR (DDETR) [38] for the problem of open-world object detection (OWOD) by introducing

所提出的OW-DETR通过引入以下部分，使标准Deformable DETR（DDETR）[38]适应开放世界目标检测(OWOD)问题：

 (i) an attention-driven pseudo-labeling mechanism (Sec. 2.3) for selecting likely unknown query candidates;

 (i) 基于注意力驱动的伪标注机制(第2.3节)，用于选择可能的未知查询候选。

 (ii) a novelty classification branch (Sec. 2.4) for learning to classify the object queries into one of the many known classes or the unknown class;

(ii) 新颖性分类分支（第2.4节），学习将对象查询分类为许多已知类别中的一类或未知类别；

 and (iii) an ‘objectness’branch (Sec. 2.5) for learning to separate the foreground objects (ground-truth known and pseudolabeled unknown instances) from the background.

(iii) “目标性”分支（第2.5节），学习将前景对象（真实已知和伪标记的未知实例）与背景分开。

 In the proposed OW-DETR, an image $I$ of spatial size $H\times W$ with a set of object instances $Y$ is input to a feature extraction backbone.

在提出的OW-DETR中，具有对象实例集合$Y$的空间尺寸为$H\times W$的图像$I$被输入到特征提取骨干网络中。

 $D$-dimensional multi-scale features are obtained at different resolutions and input to a transformer encoder-decoder containing multi-scale deformable attention modules. 

在不同分辨率下获得$D$维多尺度特征，并输入到包含多尺度可变形注意力模块的Transformer编码器-解码器中。

The decoder transforms a set of $M$ learnable object queries, aided by interleaved cross-attention and selfattention modules, to a set of $M$ object query embeddings $\boldsymbol{q}_e\in\mathbb{R}^D$ that encode potential object instances in the image.

解码器通过交错的交叉注意力和自注意力模块，将一组$M$个可学习的对象查询转换为一组编码潜在图像中对象实例的$M$个对象查询嵌入$\boldsymbol{q}_e\in\mathbb{R}^D$

###  2.1.2 第二段

The $q_e$ are then input to three branches:bounding box regression, novelty classification and objectness.

 $q_e$然后被输入到三个分支：边界框回归，新颖性分类和目标物识别。

 While the novelty classification $(F_{cls})$ and objectness $(F_{\mathrm{o}bj})$ branches
 are single layer feed-forward networks (FFN), the regression branch $F_{reg}$ is a 3-layer FFN. 

虽然新颖性分类$(F_{cls})$ 和目标物识别 $(F_{\mathrm{o}bj})$ 分支是单层前馈网络(FFN),边界框回归 $F_{reg}$ 是一个3层的前馈神经网络(FFN).

A bipartite matching loss, based on the class and box co-ordinate predictions, is employed to select unique queries that best match the ground-truth (GT) known instances.

基于类别和框坐标预测的双向匹配损失被用来选择最佳匹配真实(GT)已知实例的唯一查询。

 The remaining object queries are then utilized to select the candidate unknown class instances, which are crucial for learning in the OWOD setting.

剩余的目标查询然后被用于选择候选的未知类别实例，这对于在OWOD设置中的学习非常关键。

 To this end, an attention map $A$ obtained from the latent feature maps of the backbone is utilized to compute an objectness score $s_o$ for a query $q_e$.

为此，从骨干网络的潜在特征图中获得的注意力映射$A$被用于计算查询$q_e$的目标性分数$s_0$。

 The score $s_o$ is based on the activation magnitude inside the query's region-ofinterest in $A$. 

该分数$s_0$基于$A$中查询感兴趣区域内的激活强度。

The queries with high scores $s_o$ are selected as candidate instances and pseudo-labeled as ‘unknown’.

具有高分数$s_0$的查询被选择为候选实例，并且被伪标注为"未知"。

 These pseudo-labeled unknown queries along with the collective GT known queries are employed as foreground objects to train the objectness branch. 

这些伪标注的未知查询以及收集的真实(GT)已知查询被用作前景对象来训练目标性分支。

Moreover, while regression branch predicts the bounding box, the novelty classification branch classifies a query into one of the many known classes and an unknown class. 

此外，虽然回归分支预测边界框，但新颖性分类分支将查询分类为许多已知类别中的一类和未知类别。

The proposed OW-DETR framework is trained end-to-end using dedicated loss terms for novelty classification $(\mathcal{L}_n)$, objectness scoring $(\mathcal{L}_o)$, in addition to bounding box regression $(\mathcal{L}_r)$ in a joint formulation.

所提出的OW-DETR框架使用专门的损失项进行新颖性分类$(\mathcal{L}_n)$, 目标性得分 $(\mathcal{L}_o)$, 以及连同公式中的边界框回归 $(\mathcal{L}_r)$进行端到端的训练。

 Next, we present our OW-DETR approach in detail.

接下来，我们详细介绍我们的OW-DETR方法。



## 2.2 多尺度上下文编码Multi-scale Context Encoding

### 2.2.1 第一段

As discussed earlier in Sec. 1, given the diverse nature of unknown objects that can possibly occur in an image, detecting objects of different sizes while encoding their rich context is one of the major challenges in open-world object detection (OWOD). 

正如前面第1节所讨论的，鉴于图像中可能出现的未知目标的多样性，在编码其丰富上下文的同时检测不同大小的物体是开放世界目标检测(OWOD)的主要挑战之一。

Encoding such rich context requires capturing long-term dependencies from large receptive fields at multiple scales of the image. 

想要编码这样丰富的上下文需要从图像的多个尺度上、大的接收域上捕获长期依赖关系。

Moreover, having lesser inductive biases in the framework that make fewer assumptions about unknown objects, occurring during testing, is likely to be beneficial for improving their detection.

此外，在框架中需要只包含较少的归纳偏置，这样可以在测试过程中对未知目标做出更加少的假设。

### 2.2.2 第二段

Motivated by the above observations about OWOD task requirements, we adapt the recently introduced single-stage Deformable DETR [38] (DDETR), which is end-to-end trainable and has shown promising performance in standard object detection due to its ability to encode long-term multi-scale context with fewer inductive biases. 

受上述关于OWOD任务需求的观察结果的启发，我们采用了最近引入的单阶段 Deformable DETR(DDETR),它是端到端可训练的，并且在标准对象检测中表现出很好的性能，因为它能够以较少的归纳偏置编码长期长尺度上下文。

DDETR introduces multi-scale deformable attention modules in the transformer encoder and decoder layers of DETR [3] for encoding multi-scale context with better convergence and lower complexity. The multi-scale deformable attention module, based on deformable convolution [5, 37], only attends to a small fixed number of key sampling points around a reference point. This sampling is performed across multiscale feature maps and enables encoding richer context over a larger receptive field. For more details, we refer to [3,38].

Despite achieving promising performance for the object detection task, the standard DDETR is not suited for detecting unknown class instances in the OWOD setting. To enable detecting novel objects, we introduce an attention-driven pseudo-labeling scheme along with novelty classification and objectness branches, as explained next.