import torch
import torch.nn as nn
from transfer_losses import TransferLoss
import backbones
from metrics import *
import copy 


class TransferNet(nn.Module):
    def __init__(self, num_class, base_net='resnet50', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, metric='arc_margin', metric_s=20, metric_m=0.5, max_iter=1000, **kwargs): 
        super(TransferNet, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        self.metric = metric
        self.s = metric_s
        self.m = metric_m
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()
        
        if self.metric == 'add_margin':    
            self.classifier_layer = AddMarginProduct(feature_dim, self.num_class, self.s, self.m)  
        elif self.metric == 'arc_margin':
            self.classifier_layer = ArcMarginProduct(feature_dim, self.num_class, self.s, self.m, easy_margin=False)
        elif self.metric == 'sphere':
            self.classifier_layer = SphereProduct(feature_dim, self.num_class, self.s, self.m)
        else:
            self.classifier_layer = nn.Linear(feature_dim, self.num_classes)
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, source, target, source_label):
        source = self.base_network(source)
        target = self.base_network(target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)

        # source arc classification （L_AM）
        source_clf = self.classifier_layer(source,source_label)      
        clf_loss = self.criterion(source_clf, source_label)

        # spource target transfer （L_D）
        kwargs = {}
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target,None)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        transfer_loss = self.adapt_loss(source, target, **kwargs)


        # target center alignment loss （L_CA）
        loss_metric = torch.tensor(0.0,requires_grad=True)
        target_norm = F.normalize(target)
        target_clf = self.classifier_layer(target,None)
        target_logits = torch.nn.functional.softmax(target_clf, dim=1)
        num = 0    
        
        for i in range(64):
            if max(target_logits[i]) > 0.9:    #threshold T=0.9
                index = torch.argmax(target_logits[i])
                source_w_norm = F.normalize(self.classifier_layer.weight) 
                source_w_norm_index = source_w_norm[index]
                metric_loss_i = (F.linear(target_norm[i], source_w_norm_index))
                loss_metric = loss_metric + metric_loss_i
                num =num + 1

        metric_loss = 1-(loss_metric / (num+torch.tensor(1e-6)))

        # target entropy loss  （L_EM）
        weight_copy = copy.deepcopy(self.classifier_layer.weight)
        weight_copy.requires_grad = False
        target_clf_copy = F.linear(target, weight_copy)
        target_logits_copy = torch.nn.functional.softmax(target_clf_copy, dim=1)

        entropy_loss = -torch.mean((target_logits_copy * torch.log(target_logits_copy + 1e-6)).sum(dim=1))

        return clf_loss, transfer_loss, metric_loss, entropy_loss  

    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        features = self.base_network(x)
        x = self.bottleneck_layer(features)
        clf = self.classifier_layer(x,None) 
        return clf