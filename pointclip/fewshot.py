import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from PIL import Image
#import matplotlib.pyplot as plt

from PointDA.pointclip.clip import clip

from PointDA.pointclip.mv_utils_fs import PCViews


CUSTOM_TEMPLATES = {
    'ModelNet40': 'point cloud of a big {}.'
}


# source: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/util.py
def smooth_loss(pred, gold):
    eps = 0.2
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss = -(one_hot * log_prb).sum(dim=1).mean()
    return loss


class BatchNormPoint(nn.Module):
    def __init__(self, feat_size, sync_bn=False):
        super().__init__()
        self.feat_size = feat_size
        self.sync_bn = sync_bn
        if self.sync_bn:
            self.bn = BatchNorm2dSync(feat_size)
        else:
            self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        if self.sync_bn:
            # 4d input for BatchNorm2dSync
            x = x.view(s1 * s2, self.feat_size, 1, 1)
            x = self.bn(x)
        else:
            x = x.view(s1 * s2, self.feat_size)
            x = self.bn(x)
        return x.view(s1, s2, s3)


def load_clip_to_cpu(model):
    backbone_name = model
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    #model_path = "/home/kemove/.cache/clip/RN101.pt"

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')

    model = clip.build_model(state_dict or model.state_dict())

    return model


class PointCLIP_Model(nn.Module):

    def __init__(self, clip_model):
        super().__init__()

        # Encoders from CLIP
        self.visual_encoder = clip_model.visual

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Multi-view projection
        self.num_views =10
        pc_views = PCViews()
        self.get_img = pc_views.get_img

        self.channel = 1024


    def forward(self, pc, label=None):
        # Project to multi-view depth maps

        pc = pc.permute(0, 2, 1)
        images = self.mv_proj(pc).type(self.dtype)

        vis_imgs = images.cpu().numpy()
        # print(vis_imgs.shape)


        # for i in range(20):
        #     # print(vis_imgs[i, 0, 100, 100])
        #    vis_img = np.uint8(vis_imgs[i, :, :, :]*255).transpose(1,2,0)
        #    black_pixels = (vis_img == [0, 0, 0]).all(axis=2)
        #    vis_img[black_pixels] = [255, 255, 255]
        #    a = Image.fromarray(vis_img)
           #a.save("/home/kemove/下载/code/DefRec_and_PCM/PointDA/pointclip/abc4/%i.jpeg" % i)

              #print(a.shape)



        with torch.no_grad():
            # Image features
            image_feat = self.visual_encoder(images)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            #print(image_feat.shape)
            B_i, _ = image_feat.shape
            B = int(B_i / self.num_views)
            #print(B)
            image_feat = image_feat.view(B, self.num_views, 1024)
            image_feat = torch.max(image_feat, dim=1, keepdim=True)[0].view(B, 1024)
            #image_feat = torch.mean(image_feat, dim=1, keepdim=True).view(B, 1024)
            #self.net_2(torch.max(y, 1)[0].view(y.shape[0], -1))
            #print(image_feat.shape)
            #image_feat = image_feat.reshape(-1, self.num_views * self.channel)


        return image_feat

    def mv_proj(self, pc):
        img = self.get_img(pc).cuda()
        img = img.unsqueeze(1).repeat(1, 3, 1, 1)



        return img


# class Adapter(nn.Module):
#     """
#     Inter-view Adapter
#     """
#
#     def __init__(self):
#         super().__init__()
#
#         self.num_views = 6
#         self.in_features = 1024
#         self.adapter_ratio = 0.6
#         self.fusion_init = cfg.MODEL.ADAPTER.INIT
#         self.dropout = cfg.MODEL.ADAPTER.DROPOUT
#
#         self.fusion_ratio = nn.Parameter(torch.tensor([self.fusion_init] * self.num_views), requires_grad=True)
#
#         self.global_f = nn.Sequential(
#             BatchNormPoint(self.in_features),
#             nn.Dropout(self.dropout),
#             nn.Flatten(),
#             nn.Linear(in_features=self.in_features * self.num_views,
#                       out_features=self.in_features),
#             nn.BatchNorm1d(self.in_features),
#             nn.ReLU(),
#             nn.Dropout(self.dropout))
#
#         self.view_f = nn.Sequential(
#             nn.Linear(in_features=self.in_features,
#                       out_features=self.in_features),
#             nn.ReLU(),
#             nn.Linear(in_features=self.in_features,
#                       out_features=self.in_features * self.num_views),
#             nn.ReLU())
#
#     def forward(self, feat):
#         img_feat = feat.reshape(-1, self.num_views, self.in_features)
#         res_feat = feat.reshape(-1, self.num_views * self.in_features)
#
#         # Global feature
#         global_feat = self.global_f(img_feat * self.fusion_ratio.reshape(1, -1, 1))
#         # View-wise adapted features
#         view_feat = self.view_f(global_feat)
#
#         img_feat = view_feat * self.adapter_ratio + res_feat * (1 - self.adapter_ratio)
#
#         return img_feat


if __name__ == '__main__':
    clip_model = load_clip_to_cpu('RN50')

    model = PointCLIP_Model(clip_model)
    input = torch.autograd.Variable(torch.Tensor(16, 1024, 3))
    result = model(input)