import os

import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html, util
import numpy as np


def make_input(input):
    image_paths = input['A_paths']
    ab_constant = input['ab'].to(device)
    hist = input['hist'].to(device)

    real_A_l, real_A_ab, real_R_l, real_R_ab, real_R_histogram = [], [], [], [], []
    for i in range(3):
        real_A_l += input['A_l'][i].to(device).unsqueeze(0)
        real_A_ab += input['A_ab'][i].to(device).unsqueeze(0)
        real_R_l += input['R_l'][i].to(device).unsqueeze(0)
        real_R_ab += input['R_ab'][i].to(device).unsqueeze(0)
        real_R_histogram += [util.calc_hist(input['A_ab']
                                            [i].to(device), device)]

    return dict(
        image_paths=image_paths,
        ab_constant=ab_constant,
        hist=hist,
        real_A_l=real_A_l,
        real_A_ab=real_A_ab,
        real_R_l=real_R_l,
        real_R_ab=real_R_ab,
        real_R_histogram=real_R_histogram,
    )


if __name__ == '__main__':
    torch.set_num_threads(1)
    
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    device = torch.device('cuda:{}'.format(
        opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    scores = []
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        data = make_input(data)
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        metrics = model.compute_scores()
        scores.extend(metrics)
        print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()
    print('Histogram Intersection: %.4f' % np.mean(scores))
