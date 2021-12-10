import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.nn import Parameter
import torchvision.transforms  as transforms


class Tensorbuilter(nn.Module):

    """
    Naive tensor bank builder, with slightly lower performance

    input: attention b,Ｎ_part,w,h

           feature b,c,w,h


    return:

    tensorbank # b*c* Ｎ_part


    """
    def __init__(self):
        super(Tensorbuilter, self).__init__()

        self.alpha = Parameter(torch.zeros(1), requires_grad=True)
        self.gap_resize = nn.AdaptiveAvgPool2d(28)
        self.gap_all = nn.AdaptiveAvgPool2d(1)

    def forward(self, feature, attention):
        feature_shape = feature.size()
        attention_shape = attention.size()

        fix_w = 28

        ### spatial wise
        feature_sp = feature  # self.gap_resize(feature)
        attention_sp = attention  # self.gap_resize(attention)
        feature_sp = feature_sp.view(feature_shape[0], feature_shape[1], -1)
        attention_sp = attention_sp.view(attention_shape[0], attention_shape[1], -1)

        # print(feature_sp.size())
        phi_sp = torch.einsum('icn,icm->inm', (attention_sp, feature_sp))

        phi_sp = torch.div(phi_sp, float(attention_shape[1]))
        phi_sp = torch.mul(torch.sign(phi_sp), torch.sqrt(torch.abs(phi_sp) + 1e-12))
        phi_sp = phi_sp.view(feature_shape[0], -1)
        raw_sp = torch.nn.functional.normalize(phi_sp, dim=-1)
        sp_feature = raw_sp  #
        sp_feature = sp_feature.view(feature_shape[0], -1, feature_shape[2] * feature_shape[3])
        # print(attention.size())

        ### channel wise b,c,part_num

        en_feature = torch.einsum('icn,inm->icm', (
            attention.view(feature_shape[0], feature_shape[1], feature_shape[2] * feature_shape[3]), sp_feature))
        en_feature = en_feature.view(feature_shape[0], feature_shape[1], feature_shape[2], feature_shape[3])
        en_feature = self.alpha * en_feature + attention
        # print(self.alpha)
        # print(en_feature)

        # _,inds=self.gap_all(en_feature).topk(32,dim=1)
        #
        # g_ind= inds.expand(inds.size(0),inds.size(1),feature_shape[2],feature_shape[3])
        # attention=torch.gather(en_feature,1,g_ind)

        # print(attention[0,0,:,:])
        attention = en_feature[:, :64, :, :]

        phi = torch.einsum('imjk,injk->imn', (attention, feature))
        phi = torch.div(phi, float(attention_shape[2] * attention_shape[3]))
        phi = torch.mul(torch.sign(phi), torch.sqrt(torch.abs(phi) + 1e-12))
        phi = phi.view(feature_shape[0], -1)
        raw_feature = torch.nn.functional.normalize(phi, dim=-1)
        channel_feature = raw_feature
        # balance different input size
        # print(channel_feature[0])
        return sp_feature.view(feature_shape[0], -1, feature_shape[2], feature_shape[3]), channel_feature.view(
            feature_shape[0], feature_shape[1], -1)  # b*c* 32 part


class GatedTensorBankbuilter(nn.Module):

    """

    build gated tensor for learning, normalization funcs can be further updated for better performance.

    n_part : input part numbers default set as 512 for best performance

    alpha: learnable spatial weight, adaptive to the feature size, pooled to 28 * 28


    input: attention b,Ｎ_part,w,h

           feature b,c,w,h


    return:

    tensorbank # b*c* Ｎ_part

    """
    def __init__(self, n_part=512):
        super(GatedTensorBankbuilter, self).__init__()
        self.beta = nn.Linear(512, 1, bias=False)
        self.alpha = nn.Linear(28 * 28, 1, bias=False)
        self.gap_resize = nn.AdaptiveAvgPool2d(28)
        self.gap_all = nn.AdaptiveAvgPool2d(1)
        self.n_part = n_part

    def forward(self, feature, attention):
        feature_shape = feature.size()
        attention_shape = attention.size()

        fix_w = 28

        ### spatial wise
        feature_sp = feature  # self.gap_resize(feature)
        attention_sp = attention  # self.gap_resize(attention)
        feature_sp = feature_sp.view(feature_shape[0], feature_shape[1], -1)
        attention_sp = attention_sp.view(attention_shape[0], attention_shape[1], -1)

        # print(feature_sp.size())
        phi_sp = torch.einsum('icn,icm->inm', (attention_sp, feature_sp))

        phi_sp = torch.div(phi_sp, float(attention_shape[1]))
        phi_sp = torch.mul(torch.sign(phi_sp), torch.sqrt(torch.abs(phi_sp) + 1e-12))
        phi_sp = phi_sp.view(feature_shape[0], -1)
        # modified this to channel dimension
        raw_sp = torch.nn.functional.normalize(phi_sp, dim=-1)
        sp_feature = raw_sp  #
        sp_feature = sp_feature.view(feature_shape[0], -1, feature_shape[2] * feature_shape[3])
        # print(attention.size())

        ### channel wise b,c,part_num

        en_feature = torch.einsum('icn,inm->icm', (
            attention.view(feature_shape[0], feature_shape[1], feature_shape[2] * feature_shape[3]), sp_feature))
        en_feature = en_feature.view(feature_shape[0], feature_shape[1], feature_shape[2], feature_shape[3])

        # from spatial
        gating = self.gap_resize(en_feature)
        gating = self.alpha(gating.mean(1).view(feature_shape[0], -1))
        gating = gating.unsqueeze(1).unsqueeze(1)

        ## from channel
        # gating = self.gap_all(en_feature)
        # gating = self.beta(gating.view(feature_shape[0], -1))
        # gating = gating.unsqueeze(1).unsqueeze(1)
        ##
        en_feature = gating * en_feature + attention


        # _,inds=self.gap_all(en_feature).topk(32,dim=1)
        #
        # g_ind= inds.expand(inds.size(0),inds.size(1),feature_shape[2],feature_shape[3])
        # attention=torch.gather(en_feature,1,g_ind)

        # print(attention[0,0,:,:])
        attention = en_feature[:, :self.n_part, :, :]

        phi = torch.einsum('imjk,injk->imn', (attention, feature))
        phi = torch.div(phi, float(attention_shape[2] * attention_shape[3]))
        phi = torch.mul(torch.sign(phi), torch.sqrt(torch.abs(phi) + 1e-12))
        phi = phi.view(feature_shape[0], -1)


        # modified this to channel dimension
        raw_feature = torch.nn.functional.normalize(phi, dim=-1)
        channel_feature = raw_feature
        # balance different input size
        # print(channel_feature[0])
        return sp_feature.view(feature_shape[0], -1, feature_shape[2], feature_shape[3]), channel_feature.view(
            feature_shape[0], feature_shape[1], -1)  # b*c* Ｎ part


class ATT():
    def __init__(self):
        super(ATT, self).__init__()
        self.THRE_IOU = 0.6

    def calc_iou(self, pred, gt):
        # calc segmentation iou

        const_zero = torch.zeros(1)
        x1, x2, y1, y2 = pred

        gt_x1, gt_x2, gt_y1, gt_y2 = gt

        xA = torch.max(x1, gt_x1)
        yA = torch.max(y1, gt_y1)
        xB = torch.min(x2, gt_x2)
        yB = torch.min(y2, gt_y2)

        interArea = torch.max(const_zero, xB - xA) * torch.max(const_zero, yB - yA)

        bboxAArea = (x2 - x1) * (y2 - y1)
        bboxBArea = (gt_y2 - gt_y1) * (gt_x2 - gt_x1)

        iou = interArea / float(bboxBArea + bboxAArea - interArea)

        return iou

    def NMS_crop(self, attention_maps, input_image,nselect,ntopk):

        """
        must set this before using nms-crop

        """
        num_select = nselect #3  # 3
        num_topk = ntopk #40  # 64  # maximum N


        # start = time.time()
        B, N, W, H = input_image.shape

        input_tensor = input_image
        batch_size, num_parts, height, width = attention_maps.shape

        attention_maps = attention_maps.clone().detach()

        # vis_ret = torch.zeros(28)
        # attention_maps = torch.nn.functional.interpolate(attention_maps.detach(), size=(W, H), mode='bilinear')
        # part_weights = F.avg_pool2d(attention_maps, (W, H)).reshape(batch_size, -1)
        part_weights = attention_maps.mean(3).mean(2).reshape(batch_size, -1)
        # part_weights = attention_maps.view(batch_size,num_parts,-1)
        # part_weights,partidx = torch.max(part_weights,dim=2)
        # print(partidx)
        part_weights = torch.add(torch.sqrt(part_weights), 1e-12)
        part_weights = torch.div(part_weights, torch.sum(part_weights, dim=1).unsqueeze(1))  # .cpu()
        part_weights = part_weights  # .numpy()

        vis_ret = []
        list_img = []  # torch.empty(size=[1,batch_size,3,W//2,H//2]).cuda()

        # print(part_weights[3])
        for i in range(batch_size):
            # ret_imgs = torch.empty(size=[1,3,W//2,H//2]).cuda()
            ret_imgs = []
            vis = []

            proposals = []

            now_select = 0  # initilization important~

            attention_map = attention_maps[i]
            part_weight = part_weights[i]
            # print(part_weight.shape)
            # selected_index = np.random.choice(
            #    np.arange(0, num_parts), 1, p=part_weight)[0]
            value, idx = torch.topk(part_weight, num_topk, dim=0)
            # mask = attention_map[selected_index, :, :]

            mask = torch.index_select(attention_map, 0, idx)

            # mask = (mask-mask.min())/(mask.max()-mask.min())

            for enum in range(num_topk):
                mask[enum][0:2, :] = 0
                mask[enum][:, 0:2] = 0

                if now_select >= num_select:
                    break

                threshold = random.uniform(0.4, 0.6)

                # threshold = torch.median(mask[enum].view(-1),dim=-1)

                # threshold = 0.5
                # itemindex = torch.where(mask[enum] >= mask[enum].max() * threshold)
                # itemindex = torch.nonzero(mask >= threshold)

                if False:
                    loc = torch.where(mask[enum] != mask[enum])
                    mask[enum][loc] = 0
                ## end of nan check

                #  this sentence greatly affects the final performance.

                # mask[enum] = (mask[enum] - torch.min(mask[enum])) / (torch.max(mask[enum])-torch.min(mask[enum])+0.01)

                itemindex = torch.where(mask[enum] < mask[enum].max() * threshold)

                if itemindex[0].size(0) == 0:
                    print ("error finding")
                    new_mask = torch.ones_like(mask[enum])
                    itemindex = torch.where(new_mask == new_mask)

                padding_h = int(0.2 * H // 2)
                padding_w = int(0.2 * W // 2)

                if itemindex[0].max() - itemindex[0].min() > 4:
                    padding_h = 0

                if itemindex[1].max() - itemindex[1].min() > 4:
                    padding_w = 0

                height_min = itemindex[0].min()
                height_min = max(0, height_min - padding_h)
                height_max = (itemindex[0].max() + padding_h)
                width_min = itemindex[1].min()
                width_min = max(0, width_min - padding_w)
                width_max = (itemindex[1].max() + padding_w)

                if height_min == height_max:
                    height_min = 0
                    height_max = H
                if width_min == width_max:
                    width_min = 0
                    width_max = W

                # x1 x2,y1,y2
                bbox_tmp = torch.Tensor([width_min, width_max, height_min, height_max])
                bbox_now = bbox_tmp  # torch.from_numpy(bbox_tmp)
                #  search NMS
                flag_use = 1
                for p in proposals:
                    iou = self.calc_iou(bbox_now, p)
                    if iou > self.THRE_IOU:
                        flag_use = 0
                        break

                if flag_use == 1:
                    proposals.append(bbox_now)
                    now_select = now_select + 1
                    out_img = input_tensor[i][:, height_min:height_max, width_min:width_max].unsqueeze(0)
                    out_img = torch.nn.functional.interpolate(out_img, size=(W, H), mode='bilinear',
                                                              align_corners=True)
                    # print(proposals)
                    vis.append(mask[enum].unsqueeze(0))
                    ret_imgs.append(out_img)

            # process if not staisfiy select num
            if now_select < num_select:
                vis = []
                ret_imgs = []  # reset as zero
                # use first three part
                value, idx = torch.topk(part_weight, num_select, dim=0)

                mask = torch.index_select(attention_map, 0, idx)

                for enum in range(num_select):
                    mask[enum][0:2, :] = 0
                    mask[enum][:, 0:2] = 0
                    threshold = random.uniform(0.4, 0.6)

                    # threshold = 0.5
                    itemindex = torch.where(mask[enum] >= mask[enum].max() * threshold)

                    if itemindex[0].size(0) == 0:
                        print ("error finding inner")
                        new_mask = torch.ones_like(mask[enum])
                        itemindex = torch.where(new_mask == new_mask)

                    padding_h = int(0.2 * H // 2)
                    padding_w = int(0.2 * W // 2)

                    if itemindex[0].max() - itemindex[0].min() > 4:
                        padding_h = 0

                    if itemindex[1].max() - itemindex[1].min() > 4:
                        padding_w = 0

                    height_min = itemindex[0].min()
                    height_min = max(0, height_min - padding_h)
                    height_max = (itemindex[0].max() + padding_h)
                    width_min = itemindex[1].min()
                    width_min = max(0, width_min - padding_w)
                    width_max = (itemindex[1].max() + padding_w)

                    if height_min == height_max:
                        height_min = 0
                        height_max = H
                    if width_min == width_max:
                        width_min = 0
                        width_max = W

                    out_img = input_tensor[i][:, height_min:height_max, width_min:width_max].unsqueeze(0)
                    out_img = torch.nn.functional.interpolate(out_img, size=(W, H), mode='bilinear',
                                                              align_corners=True)
                    # print(height_min,height_max,width_min,width_max)
                    vis.append(mask[enum].unsqueeze(0))
                    ret_imgs.append(out_img)

            vis = torch.cat(vis, dim=0)
            ret_imgs = torch.cat(ret_imgs, dim=0)
            #  add batch dim

            ret_imgs = ret_imgs.unsqueeze(1)
            vis = vis.unsqueeze(1)

            list_img.append(ret_imgs)
            vis_ret.append(vis)
            # for visualization can be commented
            # if i == 0:
            #    vis_ret = vis
        # batch *3

        list_img = torch.cat(list_img, dim=1)
        vis_ret = torch.cat(vis_ret, dim=1)

        # list_img: part *batch *C*W*H

        list_img = list_img.permute(1, 0, 2, 3, 4)

        return list_img, vis_ret

    def feature_crop(self, attention_maps, input_image):

        # start = time.time()
        B, N, W, H = input_image.shape

        input_tensor = input_image
        batch_size, num_parts, height, width = attention_maps.shape

        attention_maps = attention_maps.detach()

        vis_ret = torch.zeros(28)
        # attention_maps = torch.nn.functional.interpolate(attention_maps.detach(), size=(W, H), mode='bilinear')
        # part_weights = F.avg_pool2d(attention_maps, (W, H)).reshape(batch_size, -1)
        part_weights = attention_maps.mean(3).mean(2).reshape(batch_size, -1)
        part_weights = torch.add(torch.sqrt(part_weights), 1e-12)
        part_weights = torch.div(part_weights, torch.sum(part_weights, dim=1).unsqueeze(1))  # .cpu()
        part_weights = part_weights  # .numpy()

        list_img = []  # torch.empty(size=[1,batch_size,3,W//2,H//2]).cuda()
        num_select = 3
        num_topk = 64  # maximum N
        # print(part_weights[3])
        for i in range(batch_size):
            # ret_imgs = torch.empty(size=[1,3,W//2,H//2]).cuda()
            ret_imgs = []

            attention_map = attention_maps[i]
            part_weight = part_weights[i]
            # print(part_weight.shape)
            # selected_index = np.random.choice(
            #    np.arange(0, num_parts), 1, p=part_weight)[0]
            value, idx = torch.topk(part_weight, num_select, dim=0)
            # mask = attention_map[selected_index, :, :]

            mask = torch.index_select(attention_map, 0, idx)
            vis = mask[0].clone()

            # mask = (mask-mask.min())/(mask.max()-mask.min())

            for enum in range(num_select):
                threshold = random.uniform(0.4, 0.6)

                # threshold = torch.median(mask[enum].view(-1),dim=-1)

                # threshold = 0.5
                itemindex = torch.where(mask[enum] >= mask[enum].max() * threshold)
                # itemindex = torch.nonzero(mask >= threshold)

                padding_h = int(0.2 * H // 2)
                padding_w = int(0.2 * W // 2)

                if itemindex[0].max() - itemindex[0].min() > 4:
                    padding_h = 0

                if itemindex[1].max() - itemindex[1].min() > 4:
                    padding_w = 0

                height_min = itemindex[0].min()
                height_min = max(0, height_min - padding_h)
                height_max = (itemindex[0].max() + padding_h)
                width_min = itemindex[1].min()
                width_min = max(0, width_min - padding_w)
                width_max = (itemindex[1].max() + padding_w)

                if height_min == height_max:
                    height_min = 0
                    height_max = H
                if width_min == width_max:
                    width_min = 0
                    width_max = W

                out_img = input_tensor[i][:, height_min:height_max, width_min:width_max].unsqueeze(0)
                out_img = torch.nn.functional.interpolate(out_img, size=(W, H), mode='bilinear',
                                                          align_corners=True)

                ret_imgs.append(out_img)

            ret_imgs = torch.cat(ret_imgs, dim=0)
            #  add batch dim
            ret_imgs = ret_imgs.unsqueeze(1)
            # print(ret_imgs.size())
            list_img.append(ret_imgs)

            # for visualization can be commented
            if i == 0:
                vis_ret = vis
        # batch *3
        list_img = torch.cat(list_img, dim=1)
        # print(list_img.size())
        list_img = list_img.permute(1, 0, 2, 3, 4)
        return list_img, vis_ret.unsqueeze(0)

    def attention_crop(self, attention_maps, input_image):

        # start = time.time()
        B, N, W, H = input_image.shape

        input_tensor = input_image
        batch_size, num_parts, height, width = attention_maps.shape

        attention_maps = attention_maps.detach()
        scale = 448 // 28
        # vis_ret = torch.zeros(28)
        # attention_maps = torch.nn.functional.interpolate(attention_maps.detach(), size=(W, H), mode='bilinear')
        # part_weights = F.avg_pool2d(attention_maps, (W, H)).reshape(batch_size, -1)
        part_weights = attention_maps.mean(3).mean(2).reshape(batch_size, -1)
        part_weights = torch.add(torch.sqrt(part_weights), 1e-12)
        part_weights = torch.div(part_weights, torch.sum(part_weights, dim=1).unsqueeze(1))  # .cpu()
        part_weights = part_weights  # .numpy()

        vis_ret = []
        list_img = []  # torch.empty(size=[1,batch_size,3,W//2,H//2]).cuda()
        num_select = 3

        # print(part_weights[3])
        for i in range(batch_size):
            # ret_imgs = torch.empty(size=[1,3,W//2,H//2]).cuda()
            ret_imgs = []

            attention_map = attention_maps[i]
            part_weight = part_weights[i]
            # print(part_weight.shape)
            # selected_index = np.random.choice(
            #    np.arange(0, num_parts), 1, p=part_weight)[0]
            value, idx = torch.topk(part_weight, num_select, dim=0)
            # mask = attention_map[selected_index, :, :]

            mask = torch.index_select(attention_map, 0, idx)
            vis = mask[0].clone()

            # mask = (mask-mask.min())/(mask.max()-mask.min())

            for enum in range(num_select):
                threshold = random.uniform(0.4, 0.6)
                # threshold = 0.5
                # itemindex = np.where(mask >= threshold)
                # itemindex = np.where(mask >= mask.max() * threshold)
                itemindex = torch.where(mask[enum] >= mask[enum].max() * threshold)
                # itemindex = torch.nonzero(mask >= threshold)

                padding_h = int(0.2 * H // 2)
                padding_w = int(0.2 * W // 2)

                if itemindex[0].max() - itemindex[0].min() > 4:
                    padding_h = 0

                if itemindex[1].max() - itemindex[1].min() > 4:
                    padding_w = 0

                height_min = itemindex[0].min()
                height_min = max(0, height_min * scale - padding_h)
                height_max = (itemindex[0].max() * scale + padding_h)
                width_min = itemindex[1].min()
                width_min = max(0, width_min * scale - padding_w)
                width_max = (itemindex[1].max() * scale + padding_w)

                if height_min == height_max:
                    height_min = 0
                    height_max = H
                if width_min == width_max:
                    width_min = 0
                    width_max = W

                out_img = input_tensor[i][:, height_min:height_max, width_min:width_max].unsqueeze(0)
                out_img = torch.nn.functional.interpolate(out_img, size=(W // 2, H // 2), mode='bilinear',
                                                          align_corners=True)

                ret_imgs.append(out_img)

            ret_imgs = torch.cat(ret_imgs, dim=0)
            #  add batch dim
            ret_imgs = ret_imgs.unsqueeze(1)
            # print(ret_imgs.size())
            list_img.append(ret_imgs)

            # for visualization can be commented
            # if i == 0:
            #    vis_ret = vis
        # batch *3
        list_img = torch.cat(list_img, dim=1)
        # print(list_img.size())
        return list_img, vis_ret.unsqueeze(0)

    def attention_drop(self, attention_maps, input_image):
        B, N, W, H = input_image.shape
        input_tensor = input_image
        batch_size, num_parts, height, width = attention_maps.shape
        attention_maps = torch.nn.functional.interpolate(attention_maps.detach(), size=(W, H), mode='bilinear')
        part_weights = F.avg_pool2d(attention_maps, (W, H)).reshape(batch_size, -1)
        part_weights = torch.add(torch.sqrt(part_weights), 1e-12)
        part_weights = torch.div(part_weights, torch.sum(part_weights, dim=1).unsqueeze(1)).cpu().numpy()
        # attention_maps = torch.nn.functional.interpolate(attention_maps,size=(W,H),mode='bilinear', align_corners=True)
        # print(part_weights.shape)
        masks = []
        for i in range(batch_size):
            attention_map = attention_maps[i].detach()
            part_weight = part_weights[i]
            selected_index = np.random.choice(
                np.arange(0, num_parts), 1, p=part_weight)[0]
            mask = attention_map[selected_index:selected_index + 1, :, :]

            # soft mask
            # threshold = random.uniform(0.2, 0.5)
            # threshold = 0.5
            # mask = (mask-mask.min())/(mask.max()-mask.min())
            # mask = (mask < threshold).float()
            threshold = random.uniform(0.2, 0.5)
            mask = (mask < threshold * mask.max()).float()
            masks.append(mask)
        masks = torch.stack(masks)
        # print(masks.shape)
        ret = input_tensor * masks
        return ret

    def attention_crop_drop(self, attention_maps, input_image):
        # start = time.time()
        B, N, W, H = input_image.shape
        input_tensor = input_image
        batch_size, num_parts, height, width = attention_maps.shape
        attention_maps = torch.nn.functional.interpolate(attention_maps.detach(), size=(W, H), mode='bilinear')
        part_weights = F.avg_pool2d(attention_maps.detach(), (W, H)).reshape(batch_size, -1)
        part_weights = torch.add(torch.sqrt(part_weights), 1e-12)
        part_weights = torch.div(part_weights, torch.sum(part_weights, dim=1).unsqueeze(1)).cpu()
        part_weights = part_weights.numpy()
        # print(part_weights.shape)
        ret_imgs = []
        masks = []
        # print(part_weights[3])
        for i in range(batch_size):
            attention_map = attention_maps[i]
            part_weight = part_weights[i]
            selected_index = np.random.choice(np.arange(0, num_parts), 1, p=part_weight)[0]
            selected_index2 = np.random.choice(np.arange(0, num_parts), 1, p=part_weight)[0]
            ## create crop imgs
            mask = attention_map[selected_index, :, :].cpu().numpy()
            # mask = (mask-mask.min())/(mask.max()-mask.min())
            threshold = random.uniform(0.4, 0.6)
            # threshold = 0.5
            itemindex = np.where(mask >= mask.max() * threshold)
            # print(itemindex.shape)
            # itemindex = torch.nonzero(mask >= threshold*mask.max())
            padding_h = int(0.1 * H)
            padding_w = int(0.1 * W)
            height_min = itemindex[0].min()
            height_min = max(0, height_min - padding_h)
            height_max = itemindex[0].max() + padding_h
            width_min = itemindex[1].min()
            width_min = max(0, width_min - padding_w)
            width_max = itemindex[1].max() + padding_w
            # print('numpy',height_min,height_max,width_min,width_max)
            out_img = input_tensor[i][:, height_min:height_max, width_min:width_max].unsqueeze(0)
            out_img = torch.nn.functional.interpolate(out_img, size=(W, H), mode='bilinear', align_corners=True)
            out_img = out_img.squeeze(0)
            ret_imgs.append(out_img)

            ## create drop imgs
            mask2 = attention_map[selected_index2:selected_index2 + 1, :, :]
            threshold = random.uniform(0.2, 0.5)
            mask2 = (mask2 < threshold * mask2.max()).float()
            masks.append(mask2)
        # bboxes = np.asarray(bboxes, np.float32)
        crop_imgs = torch.stack(ret_imgs)
        masks = torch.stack(masks)
        drop_imgs = input_tensor * masks
        return (crop_imgs, drop_imgs)
