import warnings
from argparse import Namespace
from time import time

import torch
from skimage import color
from torch import Tensor
from torch.cuda import amp

from util import util

from . import networks
from .base_model import BaseModel
from .losses import AFD, HistLoss, PerceptualLoss, SparseLoss


class MainStudentModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', dataset_mode='aligned')
        return parser

    def __init__(self, opt: Namespace) -> None:
        BaseModel.__init__(self, opt)
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G_student']
        self.netG_student = networks.define_G_student(
            opt.input_nc,
            opt.bias_input_nc,
            opt.value,
            self.isTrain,
            opt.norm,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )
        if self.isTrain:
            self.loss_names = ['G', 'AFD', 'L1', 'perc', 'hist', 'sparse']
            self.criterion_AFD = AFD(opt).to(self.device)
            self.criterion_L1 = torch.nn.L1Loss().to(self.device)
            self.criterion_perc = PerceptualLoss().to(self.device)
            self.criterion_hist = HistLoss().to(self.device)
            self.criterion_sparse = SparseLoss().to(self.device)

            self.optimizer_G = torch.optim.Adam(
                torch.nn.ModuleList(
                    [self.netG_student, self.criterion_AFD]
                ).parameters(),
                lr=2e-5,
                betas=(0.5, 0.99),
            )

            self.scaler = amp.GradScaler(enabled=self.opt.amp)

            self.rgb_from_xyz = torch.tensor(
                [
                    [3.24048134, -0.96925495, 0.05564664],
                    [-1.53715152, 1.87599, -0.20404134],
                    [-0.49853633, 0.04155593, 1.05731107],
                ]
            ).to(self.device)

    def set_input(self, input: dict[str, list[Tensor | str] | Tensor]) -> None:
        self.real_A_l = [A_l.to(self.device) for A_l in input['A_l']]
        self.real_R_l = input['R_l'].to(self.device)
        self.real_R_ab = [R_ab.to(self.device) for R_ab in input['R_ab']]
        self.hist = input['hist'].to(self.device)
        if not self.isTrain:
            self.image_paths = input['A_paths']
            self.real_A_rgb = input['A_rgb'].squeeze(0).cpu().numpy()
            self.real_R_rgb = input['R_rgb'].squeeze(0).cpu().numpy()
            self.real_R_histogram = util.calc_hist(input['A_ab'].to(self.device))

    def forward(self) -> None:
        if self.isTrain:
            self.feat_s: list[Tensor]
            self.feat_s, self.fake_imgs, self.confs = self.netG_student(
                self.real_A_l[-1],
                self.real_R_l,
                self.real_R_ab[0],
                self.hist,
            )
        else:
            start_time = time()
            self.fake_imgs = self.netG_student(
                self.real_A_l[-1],
                self.real_R_l,
                self.real_R_ab[0],
                self.hist,
            )
            self.netG_student_time = time() - start_time
            self.fake_R_histogram = util.calc_hist(self.fake_imgs[-1])

    def compute_losses_G(self) -> None:
        self.loss_G = 0
        self.loss_AFD = 0
        self.loss_L1 = 0
        self.loss_perc = 0
        self.loss_hist = 0
        self.loss_sparse = 0

        self.loss_AFD = self.criterion_AFD(self.feat_s, self.feat_t)
        for l, fake_img_s, fake_img_t in zip(
            self.real_A_l, self.fake_imgs, self.fake_imgs_t
        ):
            self.loss_L1 += self.criterion_L1(fake_img_s, fake_img_t)
            self.loss_perc += self.criterion_perc(
                self.lab2rgb_tensor(l, fake_img_s),
                self.lab2rgb_tensor(l, fake_img_t),
            )
            self.loss_hist += self.criterion_hist(fake_img_s, fake_img_t)
        self.loss_sparse = self.criterion_sparse(self.confs)

        self.loss_L1 /= 3
        self.loss_perc /= 3
        self.loss_hist /= 3

        self.loss_G = (
            4000 * self.loss_AFD
            + 1000 * self.loss_L1
            + 1000 * self.loss_perc
            + self.loss_hist
            + self.loss_sparse
        )

    def optimize_parameters(
        self, feat_t: list[Tensor], fake_imgs_t: list[Tensor]
    ) -> None:
        self.feat_t = feat_t
        self.fake_imgs_t = fake_imgs_t
        with amp.autocast(enabled=self.opt.amp):
            self.forward()
            self.compute_losses_G()
        self.optimizer_G.zero_grad()
        # self.loss_G.backward()
        self.scaler.scale(self.loss_G).backward()
        # self.optimizer_G.step()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()

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
