import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import VGG19_Weights, vgg19

from util import util


class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout)
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(False)

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.bn(self.linear(x)))
        return self.bn(self.linear(x))


class AFD(nn.Module):
    def __init__(self, args):
        super(AFD, self).__init__()
        self.attention = Attention(args)

    def forward(self, g_s, g_t):
        return sum(self.attention(g_s, g_t))


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.qk_dim = args.qk_dim
        self.n_t = args.n_t
        self.linear_trans_s = LinearTransformStudent(args)
        self.linear_trans_t = LinearTransformTeacher(args)

        self.p_t = nn.Parameter(torch.Tensor(len(args.t_shapes), args.qk_dim))
        self.p_s = nn.Parameter(torch.Tensor(len(args.s_shapes), args.qk_dim))
        torch.nn.init.xavier_normal_(self.p_t)
        torch.nn.init.xavier_normal_(self.p_s)

    def forward(self, g_s, g_t):
        bilinear_key, h_hat_s_all = self.linear_trans_s(g_s)
        query, h_t_all = self.linear_trans_t(g_t)

        p_logit = torch.matmul(self.p_t, self.p_s.t())

        logit = torch.add(
            torch.einsum('bstq,btq->bts', bilinear_key, query), p_logit
        ) / np.sqrt(self.qk_dim)
        atts = F.softmax(logit, dim=2)  # b x t x s
        loss = []

        for i, (n, h_t) in enumerate(zip(self.n_t, h_t_all)):
            h_hat_s = h_hat_s_all[n]
            diff = self.cal_diff(h_hat_s, h_t, atts[:, i])
            loss.append(diff)
        return loss

    def cal_diff(self, v_s, v_t, att):
        diff = (v_s - v_t.unsqueeze(1)).pow(2).mean(2)
        diff = torch.mul(diff, att).sum(1).mean()
        return diff


class LinearTransformTeacher(nn.Module):
    def __init__(self, args):
        super(LinearTransformTeacher, self).__init__()
        self.query_layer = nn.ModuleList(
            [nn_bn_relu(t_shape[1], args.qk_dim) for t_shape in args.t_shapes]
        )

    def forward(self, g_t):
        bs = g_t[0].size(0)
        channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]
        spatial_mean = [f_t.pow(2).mean(1).view(bs, -1) for f_t in g_t]
        query = torch.stack(
            [
                query_layer(f_t, relu=False)
                for f_t, query_layer in zip(channel_mean, self.query_layer)
            ],
            dim=1,
        )
        value = [F.normalize(f_s, dim=1) for f_s in spatial_mean]
        return query, value


class LinearTransformStudent(nn.Module):
    def __init__(self, args):
        super(LinearTransformStudent, self).__init__()
        self.t = len(args.t_shapes)
        self.s = len(args.s_shapes)
        self.qk_dim = args.qk_dim
        self.relu = nn.ReLU(inplace=False)
        self.samplers = nn.ModuleList(
            [Sample(t_shape) for t_shape in args.unique_t_shapes]
        )

        self.key_layer = nn.ModuleList(
            [nn_bn_relu(s_shape[1], self.qk_dim) for s_shape in args.s_shapes]
        )
        self.bilinear = nn_bn_relu(args.qk_dim, args.qk_dim * len(args.t_shapes))

    def forward(self, g_s):
        bs = g_s[0].size(0)
        channel_mean = [f_s.mean(3).mean(2) for f_s in g_s]
        spatial_mean = [sampler(g_s, bs) for sampler in self.samplers]

        key = torch.stack(
            [key_layer(f_s) for key_layer, f_s in zip(self.key_layer, channel_mean)],
            dim=1,
        ).view(
            bs * self.s, -1
        )  # Bs x h
        bilinear_key = self.bilinear(key, relu=False).view(bs, self.s, self.t, -1)
        value = [F.normalize(s_m, dim=2) for s_m in spatial_mean]
        return bilinear_key, value


class Sample(nn.Module):
    def __init__(self, t_shape):
        super(Sample, self).__init__()
        t_N, t_C, t_H, t_W = t_shape
        self.sample = nn.AdaptiveAvgPool2d((t_H, t_W))

    def forward(self, g_s, bs):
        g_s = torch.stack(
            [self.sample(f_s.pow(2).mean(1, keepdim=True)).view(bs, -1) for f_s in g_s],
            dim=1,
        )
        return g_s


class MyVGG(nn.Module):
    def __init__(self, features):
        super(MyVGG, self).__init__()
        self.features = features

    def forward(self, x):
        return self.features(x)


class PerceptualLoss(nn.Module):
    def __init__(self, device) -> None:
        super(PerceptualLoss, self).__init__()
        self.vggnet = MyVGG(vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:12])
        self.vggnet.eval()
        for param in self.vggnet.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()

        self.rgb_from_xyz = torch.tensor(
            [
                [3.24048134, -0.96925495, 0.05564664],
                [-1.53715152, 1.87599, -0.20404134],
                [-0.49853633, 0.04155593, 1.05731107],
            ]
        ).to(device)

    def forward(self, L, fake_img_s, fake_img_t):
        rgb_s = self.lab2rgb_tensor(L, fake_img_s)
        rgb_t = self.lab2rgb_tensor(L, fake_img_t)
        f_s = self.vggnet(rgb_s)
        f_t = self.vggnet(rgb_t)
        return self.mse_loss(f_s, f_t) / f_s.shape[1] / f_s.shape[2] / f_s.shape[3]

    def lab2rgb_tensor(self, L: Tensor, ab: Tensor):
        # Lab处理
        L = (L + 1.0) * 50.0
        ab = ab * 110.0

        # Lab分开
        L = L.permute(0, 2, 3, 1)[..., 0]
        ab = ab.permute(0, 2, 3, 1)
        a, b = ab[..., 0], ab[..., 1]

        # Lab->xyz
        y = (L + 16) / 116
        x = (a / 500) + y
        z = y - (b / 200)

        # z处理
        z[z.data < 0] = 0

        # xyz合并
        xyz = torch.stack([x, y, z], dim=-1)

        # xyz处理
        mask = xyz > 0.2068966
        xyz[mask] = torch.pow(xyz[mask], 3)
        xyz[~mask] = (xyz[~mask] - 16.0 / 116.0) / 7.787
        xyz[..., 0] *= 0.95047
        xyz[..., 2] *= 1.08883

        # xyz->rgb
        rgb = xyz @ self.rgb_from_xyz

        # rgb处理
        mask = rgb > 0.0031308
        rgb[mask] = 1.055 * torch.pow(rgb[mask], 1 / 2.4).type_as(rgb) - 0.055
        rgb[~mask] *= 12.92
        rgb[rgb < 0] = 0
        rgb[rgb > 1] = 1
        rgb *= 255

        return rgb.permute(0, 3, 1, 2)


class SparseLoss(nn.Module):
    def __init__(self) -> None:
        super(SparseLoss, self).__init__()

    def forward(self, confs):
        return sum(torch.mean(-conf * torch.log(conf)) for conf in confs)


class HistLoss(nn.Module):
    def __init__(self) -> None:
        super(HistLoss, self).__init__()

    def forward(self, fake_img_s, fake_img_t):
        TH = util.calc_hist(fake_img_s) + 1e-16
        RH = util.calc_hist(fake_img_t) + 1e-16
        return 2 * torch.sum((TH - RH) ** 2 / (TH + RH))
