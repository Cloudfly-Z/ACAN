# ACAN: A plug-and-play Adaptive Center-Aligned Network for unsupervised domain adaptation
Official implementation of [ACAN](https://doi.org/10.1016/j.engappai.2024.109132) (EAAI 2024)

Abstract
---
Domain adaptation is an important topic due to its capability in transferring knowledge from source domain to target domain. However, many existing domain adaptation methods primarily concentrate on aligning the data distributions between the source and target domains, often neglecting discriminative feature learning. As a result, target samples with low confidence are embedded near the decision boundary, where they are susceptible to being misclassified, resulting in negative transfer. To address this problem, a novel Adaptive Center-Aligned Network dubbed ACAN is proposed for unsupervised domain adaptation in this work. The main innovations of ACAN are fourfold. Firstly, it is a plug-and-play module and can be easily incorporated into any domain alignment methods without increasing the model complexity and computational burden. Secondly, in contrast to conventional softmax plus cross-entropy loss, angular margin loss is called to enhance the discrimination power for classifier. Thirdly, entropy regularization is exploited to highlight the probability of potential related class, which renders our learned feature representation far away from the decision boundary. Fourthly, to improve the discriminative capacity of model to the target domain, we propose to align the target domain samples to the corresponding class center via pseudo labels. Incorporating ACAN, the performance of baseline domain alignment methods is significantly improved. Extensive ablation and comparison experiments on four widely adopted databases demonstrate the effectiveness of our ACAN.

Motivation
---
![Motivation](fig/FigToyexample.pdf "Toy Example")

Network Architecture
---

Visualization
---
