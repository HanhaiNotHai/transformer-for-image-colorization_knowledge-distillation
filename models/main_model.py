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
        self.netG = networks.define_G(opt.input_nc, opt.bias_input_nc, opt.output_nc, opt.norm, opt.init_type,
                                      opt.init_gain, self.gpu_ids)
        self.convert = util.Convert(self.device)

    def set_input(self, input):
        self.image_paths = input['image_paths']
        self.ab_constant = input['ab_constant']
        self.hist = input['hist']
        self.real_A_l = input['real_A_l']
        self.real_A_ab = input['real_A_ab']
        self.real_R_l = input['real_R_l']
        self.real_R_ab = input['real_R_ab']
        self.real_R_histogram = input['real_R_histogram']

    def forward(self):
        start_time = time()
        self.feat_t: list[Tensor]
        self.feat_t, self.fake_imgs = self.netG(
            self.real_A_l[-1], self.real_R_l[-1], self.real_R_ab[0],
            self.hist, self.ab_constant, self.device
        )
        self.netG_time = time() - start_time
        self.fake_R_histogram = []
        for i in range(3):
            self.fake_R_histogram += [util.calc_hist(self.fake_imgs[i], self.device)]
