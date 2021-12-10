import torch
import torch.nn as nn
from .backbones.resnet import ResNet, BasicBlock, Bottleneck, resnet50_bap,resnet101_bap
from loss.arcface import ArcFace
from model.trans_utils.transformer import build_transformer
from model.trans_utils.position_encoding import build_position_encoding
from model.sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm1d
from model.backbones.Attention import *


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
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    """
          default use learning backbone network

          set the configurations in /config

    personally enable the part modules


    """


    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.LAST_STRIDE
        model_path = cfg.PRETRAIN_PATH
        self.cos_layer = cfg.COS_LAYER
        model_name = cfg.MODEL_NAME
        pretrain_choice = cfg.PRETRAIN_CHOICE

        self.nselect = cfg.NUM_SELECT_PART
        self.ntopk = cfg.NUM_PART_STACK

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = resnet50_bap(pretrained=True)
        elif model_name == 'resnet101':
            self.in_planes = 2048
            self.base = resnet101_bap(pretrained=True)
            # self.base = ResNet(last_stride=last_stride,
            #                   block=Bottleneck,
            #                   layers=[3, 4, 6, 3])
        else:
            print('unsupported backbone! only support resnet50, and resnet 101 , but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            pass
            # self.base.load_param(model_path)
            # print('Loading pretrained ImageNet model......')

        self.transdim = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, padding=0, bias=True),
            SynchronizedBatchNorm2d(256, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.ATT = ATT()

        hidden_dim = 512
        num_queries = 1
        self.query_embed = nn.Embedding(num_queries, hidden_dim)


        # mannually defined part models. can be optimized with iteration definations
        self.transformer_p4 = build_transformer(d_model=512, nhead=1, num_encoder_layers=1)
        self.transformer_p3 = build_transformer(d_model=512, nhead=1, num_encoder_layers=1)
        self.transformer_p2 = build_transformer(d_model=512, nhead=1, num_encoder_layers=1)
        self.transformer_p1 = build_transformer(d_model=512, nhead=1, num_encoder_layers=1)
        self.transformer_g = build_transformer(d_model=512, nhead=4, num_encoder_layers=3)

        # use the default sine positional embedding change the mode as sine
        #self.positional_embedding = build_position_encoding(hidden_dim=hidden_dim, mode='sine')
        self.positional_embedding = build_position_encoding(hidden_dim=hidden_dim, mode='learned')

        self.positional_embedding1 = build_position_encoding(hidden_dim=hidden_dim, mode='learned')

        self.positional_embedding2 = build_position_encoding(hidden_dim=hidden_dim, mode='learned')

        self.positional_embedding3 = build_position_encoding(hidden_dim=hidden_dim, mode='learned')

        self.positional_embedding4 = build_position_encoding(hidden_dim=hidden_dim, mode='learned')


        self.bottleneck = nn.BatchNorm1d(self.in_planes // 4)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck1 = nn.BatchNorm1d(self.in_planes // 4)
        self.bottleneck1.bias.requires_grad_(False)
        self.bottleneck1.apply(weights_init_kaiming)

        self.bottleneck2 = nn.BatchNorm1d(self.in_planes // 4)
        self.bottleneck2.bias.requires_grad_(False)
        self.bottleneck2.apply(weights_init_kaiming)

        self.bottleneck3 = nn.BatchNorm1d(self.in_planes // 4)
        self.bottleneck3.bias.requires_grad_(False)
        self.bottleneck3.apply(weights_init_kaiming)

        self.bottleneck4 = nn.BatchNorm1d(self.in_planes // 4)
        self.bottleneck4.bias.requires_grad_(False)
        self.bottleneck4.apply(weights_init_kaiming)

        if self.cos_layer:
            print('using cosine layer')
            self.arcface = ArcFace(self.in_planes, self.num_classes, s=30.0, m=0.50)
        else:
            self.classifier = nn.Linear(self.in_planes // 4, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

            self.classifier1 = nn.Linear(self.in_planes // 4, self.num_classes, bias=False)
            self.classifier1.apply(weights_init_classifier)

            self.classifier2 = nn.Linear(self.in_planes // 4, self.num_classes, bias=False)
            self.classifier2.apply(weights_init_classifier)

            self.classifier3 = nn.Linear(self.in_planes // 4, self.num_classes, bias=False)
            self.classifier3.apply(weights_init_classifier)

            self.classifier4 = nn.Linear(self.in_planes // 4, self.num_classes, bias=False)
            self.classifier4.apply(weights_init_classifier)

    def forward(self, inputs, label=None, mode='train'):
        """
        label is unused if self.cos_layer == 'no'
        Args:
            inputs: input image batched
            label: labels batched
            mode: train or test

        Returns:
            classification probabilities batched.
        """

        if mode == 'train':
            return self.forward_multi(inputs, label)
        else:
            return self.test_multi(inputs)

    def forward_onepart(self, x, label=None):  # label is unused if self.cos_layer ==

        # x = self.base(x)
        attention, feature, out = self.base(x)
        # x = self.transdim(x)

        b, c, w, h = out.size()

        mask_g = torch.zeros_like(out[:, 0, :, :], dtype=torch.bool).cuda()
        pos_g = self.positional_embedding(feature, mask_g)

        _, feat_g = self.transformer_g(feature, mask_g, self.query_embed.weight, pos_g)

        global_feat = nn.functional.avg_pool2d(feat_g, feat_g.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        # feat = feat.mean(0).view(b,-1)
        feat = self.bottleneck(global_feat)

        cls_score = self.classifier(feat)
        return cls_score  # global feature for triplet loss

    def test_multi(self, x, label=None):  # label is unused

        attention, feature, out = self.base(x)


        b, c, w, h = out.size()


        mask_g = torch.zeros_like(out[:, 0, :, :], dtype=torch.bool).cuda()
        pos_g = self.positional_embedding(feature, mask_g)

        _, feat_g = self.transformer_g(feature, mask_g, self.query_embed.weight, pos_g)



        # print (feat.size())

        global_feat = nn.functional.avg_pool2d(feat_g, feat_g.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        # feat = feat.mean(0).view(b,-1)
        feat = self.bottleneck(global_feat)

        cls_score = self.classifier(feat)
        return cls_score

    def forward_multi(self, x, label=None):  # label is unused if self.cos_layer ==

        # x = self.base(x)
        attention, feature, out = self.base(x)
        # x = self.transdim(x)

        b, c, w, h = out.size()

        _, vis_ret = self.ATT.NMS_crop(attention, feature,self.nselect,self.ntopk)




        mask_g = torch.zeros_like(out[:, 0, :, :], dtype=torch.bool).cuda()
        pos_g = self.positional_embedding(feature, mask_g)

        _, feat_g = self.transformer_g(feature, mask_g, self.query_embed.weight, pos_g)

        part_1 = vis_ret[0, :, :, :]
        mask_p1 = part_1 < part_1.mean()
        pos_p1 = self.positional_embedding1(feature, mask_p1)
        _, feat_p1 = self.transformer_p1(feature, mask_p1, self.query_embed.weight, pos_p1)

        part_2 = vis_ret[1, :, :, :]
        mask_p2 = part_2 < part_2.mean()
        pos_p2 = self.positional_embedding2(feature, mask_p2)
        _, feat_p2 = self.transformer_p2(feature, mask_p2, self.query_embed.weight, pos_p2)

        part_3 = vis_ret[2, :, :, :]
        mask_p3 = part_3 < part_3.mean()
        pos_p3 = self.positional_embedding3(feature, mask_p3)
        _, feat_p3 = self.transformer_p3(feature, mask_p3, self.query_embed.weight, pos_p3)

        part_4 = vis_ret[3, :, :, :]
        mask_p4 = part_4 < part_4.mean()
        pos_p4 = self.positional_embedding4(feature, mask_p4)
        _, feat_p4 = self.transformer_p3(feature, mask_p4, self.query_embed.weight, pos_p4)

        # print (feat.size())

        global_feat = nn.functional.avg_pool2d(feat_g, feat_g.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        # feat = feat.mean(0).view(b,-1)
        feat = self.bottleneck(global_feat)

        cls_1 = self.gap(feat_p1).view(b, -1)
        cls_1 = self.bottleneck1(cls_1)
        cls_1 = self.classifier1(cls_1)

        cls_2 = self.gap(feat_p2).view(b, -1)
        cls_2 = self.bottleneck2(cls_2)
        cls_2 = self.classifier2(cls_2)

        cls_3 = self.gap(feat_p3).view(b, -1)
        cls_3 = self.bottleneck3(cls_3)
        cls_3 = self.classifier3(cls_3)

        cls_4 = self.gap(feat_p4).view(b, -1)
        cls_4 = self.bottleneck4(cls_4)
        cls_4 = self.classifier4(cls_4)

        cls_score = self.classifier(feat)
        return cls_score, cls_1, cls_2, cls_3, cls_4

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        self.load_state_dict(param_dict)

        print('Loading pretrained model from {}'.format(trained_path))


def make_model(cfg, num_class):
    model = Backbone(num_class, cfg)
    return model
