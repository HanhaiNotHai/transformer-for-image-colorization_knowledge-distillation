from argparse import Namespace
from time import time

import torch
from torch import Tensor
from torch.cuda import amp

from util import util

from . import networks
from .base_model import BaseModel
from .losses import AFD, PerceptualLoss


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
            opt.output_nc,
            opt.norm,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )
        self.convert = util.Convert(self.device)
        if self.isTrain:
            self.loss_names = ['AFD', 'L1', 'perc']
            self.criterion_AFD = AFD(opt).to(self.device)
            self.criterion_L1 = torch.nn.L1Loss().to(self.device)
            self.criterion_perc = PerceptualLoss().to(self.device)
            self.criterion_GAN = ...
            self.criterion_sparse = ...
            self.criterion_hist = ...

            self.optimizer_G = torch.optim.Adam(
                self.netG_student.parameters(), lr=2e-5, betas=(0.5, 0.99)
            )

            self.scaler = amp.GradScaler(enabled=self.opt.amp)

    def set_input(self, input: dict[str, list[Tensor | str] | Tensor]) -> None:
        self.image_paths = input['image_paths']
        self.ab_constant = input['ab_constant']
        self.hist = input['hist']
        self.real_A_l = input['real_A_l']
        self.real_A_ab = input['real_A_ab']
        self.real_R_l = input['real_R_l']
        self.real_R_ab = input['real_R_ab']
        self.real_R_histogram = input['real_R_histogram']

    def forward(self) -> None:
        start_time = time()
        self.feat_s: list[Tensor]
        self.feat_s, self.fake_imgs = self.netG_student(
            self.real_A_l[-1],
            self.real_R_l[-1],
            self.real_R_ab[0],
            self.hist,
            self.ab_constant,
            self.device,
        )
        self.netG_student_time = time() - start_time
        self.fake_R_histogram: list[Tensor] = []
        for i in range(3):
            self.fake_R_histogram += [util.calc_hist(self.fake_imgs[i], self.device)]

    def compute_losses_G(self) -> None:
        self.loss_AFD = self.criterion_AFD(self.feat_s, self.feat_t)
        self.loss_L1 = sum(
            self.criterion_L1(f_s, f_t)
            for f_s, f_t in zip(self.fake_imgs, self.fake_imgs_t)
        )
        self.loss_perc = sum(
            self.criterion_perc(l, f_s, f_t)
            for l, f_s, f_t in zip(self.real_A_l, self.fake_imgs, self.fake_imgs_t)
        )
        self.loss_GAN = 0
        # self.loss_GAN = self.criterion_GAN()
        self.loss_sparse = 0
        # self.loss_sparse = self.criterion_sparse()
        self.loss_hist = 0
        # self.loss_hist = self.criterion_hist()
        self.loss_G = 200 * self.loss_AFD + 0.9 * (
            1000 * self.loss_L1
            + 1000 * self.loss_perc
            + 0.1 * self.loss_GAN
            + 1 * self.loss_sparse
            + 1 * self.loss_hist
        )

    def optimize_parameters(
        self, feat_t: list[Tensor], fake_imgs_t: list[Tensor]
    ) -> None:
        self.feat_t = feat_t
        self.fake_imgs_t = fake_imgs_t
        with amp.autocast(enabled=self.opt.amp):
            self.forward()
            self.compute_losses_G()
        # self.loss_G.backward()
        self.scaler.scale(self.loss_G).backward()
        # self.optimizer_G.step()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()
        self.optimizer_G.zero_grad()
