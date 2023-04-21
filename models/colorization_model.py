import cv2
import numpy as np
import torch
from skimage import color

from .main_model import MainModel


class ColorizationModel(MainModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        MainModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataset_mode='colorization')
        return parser

    def __init__(self, opt):
        MainModel.__init__(self, opt)
        self.visual_names = ['real_A_rgb', 'real_A_l_0', 'fake_R_rgb', 'real_R_rgb']

    def lab2rgb(self, L, AB):
        AB2 = AB * 110.0
        L2 = (L + 1.0) * 50.0
        Lab = torch.cat([L2, AB2], dim=1)
        Lab = Lab[0].data.cpu().float().numpy()
        Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
        rgb = color.lab2rgb(Lab) * 255
        return rgb

    def tensor2gray(self, im):
        im = im[0].data.cpu().float().numpy()
        im = np.transpose(im.astype(np.float64), (1, 2, 0))
        im = np.repeat(im, 3, axis=-1) * 255
        return im

    def compute_visuals(self):
        self.real_A_l_0 = self.real_A_l[-1]
        self.real_R_rgb = cv2.resize(
            self.real_R_rgb, (self.real_A_rgb.shape[1], self.real_A_rgb.shape[0])
        )
        self.fake_R_rgb = []
        for i in range(3):
            self.fake_R_rgb += [self.lab2rgb(self.real_A_l[i], self.fake_imgs[i])]
            if i != 2:
                self.fake_R_rgb[i] = cv2.resize(
                    self.fake_R_rgb[i],
                    (self.real_A_rgb.shape[1], self.real_A_rgb.shape[0]),
                )

    def compute_scores(self):
        hr = self.real_R_histogram.data.cpu().float().numpy().flatten()
        hg = self.fake_R_histogram.data.cpu().float().numpy().flatten()
        return cv2.compareHist(hr, hg, cv2.HISTCMP_INTERSECT)
