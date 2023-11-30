# Deformable DETR: Deformable Transformers for end-to-end Object Detection

Deformable DETR:用于端到端目标检测的Deformable Transformer

## 0、概述Abstract

DETR has been recently proposed to eliminate the need for many hand-designed components in object detection while demonstrating good performance. 

最近提出的DETR在展示良好性能的同时，消除了对许多手工设计的目标检测组件的需要。

However, it suffers from slow convergence and limited feature spatial resolution, due to the limitation of Transformer attention modules in processing image feature maps. 

然而，DETR的收敛速度慢，特征空间的分辨率有限，这些缺点来源于Transformer注意力模块在处理图像特征映射时的局限性。

To mitigate these issues, we proposed Deformable DETR, whose attention modules only attend to a small set of key sampling points around a reference.

为了缓解这些问题，我们提出了Deformable DETR，其注意力模块只关注参考周围的一小部分关键采样点。

 Deformable DETR can achieve better performance than DETR (especially on small objects) with 10× less training epochs. 

Deformable DETR可以比DETR获得更好的性能(特别是在小物体上)，训练次数减少10倍。

Extensive experiments on the COCO benchmark demonstrate the effectiveness of our approach. 

在COCO数据集基准上的大量实验证明了我们的方法的有效性。

Code is released at https:// github.com/fundamentalvision/Deformable-DETR.

略。

## 1、介绍Introduction

### 1.1 第一段

Modern object detectors employ many hand-crafted components (Liu et al, 2020), e.g., anchor generation, rule-based training target assignment, non-maximum suppression (NMS) post-processing.

现代目标检测器采用很多手工制作的组件(Liu et al,2020)，例如，anchor生成、基于规则的训练目标分配、非极大值抑制(NMS)后处理。

They are not fully end-to-end. 

他们不是完全端到端的。

Recently, Carion et al (2020) proposed DETR to eliminate the need for such hand-crafted components, and built the first fully end-to-end object detector, achieving very competitive performance.

最近，Carion等人(2020)提出了DETR，以消除对这种手工制作组件的需求，并构建了第一个完全端到端的目标检测器，实现了非常有竞争力的性能。

 DETR utilizes a simple architecture, by combining convolutional neural networks (CNNs) and Transformer (Vaswani et al, 2017) encoder-decoders.

通过将卷积神经网络(CNNs)和Transformer(Vaswani et al,2017)编码器-解码器相结合

 They exploit the versatile and powerful relation modeling capability of Transformers to replace the hand-crafted rules, under properly designed training signals.

### 1.2 第二段

Despite its interesting design and good performance, DETR has its own issues: 

尽管DETR有有趣的设计和良好的性能，但它也有自己的问题：

(1) It requires much longer training epochs to converge than the existing object detectors.

(1)与现有的目标检测器相比，它需要更长的训练时间来收敛。

 For example, on the COCO (Lin et al, 2014) benchmark, DETR needs 500 epochs to converge, which is around 10 to 20 times slower than Faster R-CNN (Ren et al, 2015). 

例如，在COCO(Lin et al.,2014)基准数据集上，DETR需要500个epoch才能收敛，比Faster R-CNN(Ren et al .,2015)慢10到20倍左右。

(2) DETR delivers relatively low performance at detecting small objects. 

(2)DETR在检测小物体时性能相对较低。

Modern object detectors usually exploit multi-scale features, where small objects are detected from high-resolution feature maps. 

现代目标检测器通常利用多尺度特征，从高分辨率特征图中检测小目标。

Meanwhile, high-resolution feature maps lead to unacceptable complexities for DETR. 

同时，高分辨率的特征映射导致了DETR不可接受的复杂性。

The above-mentioned issues can be mainly attributed to the deficit of Transformer components in processing image feature maps. 

上述问题的主要原因是Transformer组件在处理图像特征映射时存在缺陷。

At initialization, the attention modules cast nearly uniform attention weights to all the pixels in the feature maps. 

在初始化时，注意模块对特征映射中

Long training epoches is necessary for the attention weights to be learned to focus on sparse meaningful locations. On the other hand, the attention weights computation in Transformer encoder is of quadratic computation w.r.t. pixel numbers. Thus, it is of very high computational and memory complexities to process high-resolution feature maps.

