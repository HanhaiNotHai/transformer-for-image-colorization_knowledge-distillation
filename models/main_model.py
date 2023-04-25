from time import time

from torch import Tensor

from util import util

from . import networks
from .base_model import BaseModel


class MainModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', dataset_mode='aligned')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G']
        self.netG = networks.define_G(
            opt.input_nc,
            opt.bias_input_nc,
            opt.value,
            opt.norm,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )

    def set_input(self, input):
        self.real_A_l = [A_l.to(self.device) for A_l in input['A_l']]
        self.real_R_l = input['R_l'].to(self.device)
        self.real_R_ab = [R_ab.to(self.device) for R_ab in input['R_ab']]
        self.hist = input['hist'].to(self.device)
        if not self.isTrain:
            self.image_paths = input['A_paths']
            self.real_A_rgb = input['A_rgb'].squeeze(0).cpu().numpy()
            self.real_R_rgb = input['R_rgb'].squeeze(0).cpu().numpy()
            self.real_R_histogram = util.calc_hist(input['A_ab'].to(self.device))

    def forward(self):
        start_time = time()
        self.feat_t: list[Tensor]
        self.feat_t, self.fake_imgs = self.netG(
            self.real_A_l[-1],
            self.real_R_l,
            self.real_R_ab[0],
            self.hist,
        )
        self.netG_time = time() - start_time
        self.fake_R_histogram = util.calc_hist(self.fake_imgs[-1])

        return self.feat_t, self.fake_imgs
