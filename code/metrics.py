from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=20.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))   # n×d 
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if label == None:
            return cosine*self.s

        else:   
            # 由 cosθ 计算相应的 sinθ
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            # 展开计算 phi=cos(θ+m) = cosθ*cosm - sinθ*sinm, 其中包含了 Target Logit (cos(θyi+ m)) (由于输入特征 xi 的非真实类也参与了计算, 最后计算新 Logit 时需使用 One-Hot 区别)
            phi = cosine * self.cos_m - sine * self.sin_m  
            if self.easy_margin:  # 是否松弛约束??
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            # --------------------------- convert label to one-hot ---------------------------
            # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
            one_hot = torch.zeros(cosine.size(), device='cuda') # requires_grad=False
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)    
            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            # - 只有输入特征 xi 对应的真实类别 yi (one_hot=1) 采用新 Target Logit cos(θ_yi + m)
            # - 其余并不对应输入特征 xi 的真实类别的类 (one_hot=0) 则仍保持原 Logit cosθ_j
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
            output *= self.s

            return output


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if label == None:
            return cosine*self.s
        
        else:
            phi = cosine - self.m
        	# --------------------------- convert label to one-hot ---------------------------
            one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
            output *= self.s

            return output
  
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, s=20,m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))

        if label ==None:
            return cos_theta*self.s
        
        else:    
            cos_theta = cos_theta.clamp(-1, 1)
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = cos_theta.data.acos()
            k = (self.m * theta / 3.14159265).floor()
            phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
            NormOfFeature = torch.norm(input, 2, 1)

	# --------------------------- convert label to one-hot ---------------------------
            one_hot = torch.zeros(cos_theta.size())
            one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
            one_hot.scatter_(1, label.view(-1, 1), 1)

	# --------------------------- Calculate output ---------------------------
            output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
            output *= NormOfFeature.view(-1, 1)

            return output*self.s

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'
