import os

import numpy as np
import torch

from data import create_dataset
from models import create_model
from options.test_student_options import TestStudentOptions
from util import html
from util.visualizer import save_images

if __name__ == '__main__':
    torch.set_num_threads(20)

    opt = TestStudentOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.epoch='best'

    device = (
        torch.device('cuda:{}'.format(opt.gpu_ids[0]))
        if opt.gpu_ids
        else torch.device('cpu')
    )

    opt.value = (
        torch.tensor(np.load('./doc/ab_constant_filter.npy'))
        .unsqueeze(0)
        .repeat(opt.batch_size, 1, 1)
        .float()
        .to(device)
    )

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    webpage = html.HTML(
        web_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}'
    )
    scores = []
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        print('processing (%04d)-th image... %s' % (i, data['A_paths']))
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        scores.append(model.compute_scores())
        save_images(
            webpage,
            visuals,
            img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize,
        )
    webpage.save()
    print('Histogram Intersection: %.4f' % np.mean(scores))
