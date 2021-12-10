import torch
import torch.nn as nn
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from loss.arcface import ArcFace


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.LAST_STRIDE
        model_path = cfg.PRETRAIN_PATH
        self.cos_layer = cfg.COS_LAYER
        model_name = cfg.MODEL_NAME
        pretrain_choice = cfg.PRETRAIN_CHOICE
        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])

        else:
            print('unsupported backbone! only support resnet50, but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        hidden =512
        self.trans = nn.Conv2d(2048, 512, 1, 1, padding=0)
        self.bn = nn.BatchNorm2d(hidden)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        if self.cos_layer:
            print('using cosine layer')
            self.arcface = ArcFace(self.in_planes, self.num_classes, s=30.0, m=0.50)
        else:
            self.classifier = nn.Linear(512*512, self.num_classes, bias=True)
            self.classifier.apply(weights_init_classifier)



    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        #global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        #global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)


        x = self.trans(x)
        b, c, w, h = x.size()
        #x1 = self.bn(x).view(b,c,-1)
        x1 = x.view(b, c, -1)
        x2 = x1.permute([0,2,1])

        matrix =torch.matmul(x1,x2).view(b,-1)/(28*28)




        #feat = self.bottleneck(global_feat)



        if self.training:
            if self.cos_layer:
                assert False
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(matrix)
            return cls_score, matrix  # global feature for triplet loss
        else:
            if self.cos_layer:
                assert False
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(matrix)
            return cls_score, matrix  # global feature for triplet loss
            #return feat


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        self.load_state_dict(param_dict)
        # for i in param_dict:
        #     if 'classifier' in i or 'arcface' in i:
        #         continue
        #     self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

def make_model(cfg, num_class):
    model = Backbone(num_class, cfg)
    return model
