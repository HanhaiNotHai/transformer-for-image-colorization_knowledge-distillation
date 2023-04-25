import os

import numpy as np
import torch
from torch.cuda import amp

from data import create_dataset
from models import create_model
from models.colorization_model import ColorizationModel
from models.colorization_student_model import ColorizationStudentModel
from options.test_student_options import TestStudentOptions
from util import html
from util.visualizer import save_images_st

if __name__ == '__main__':
    torch.set_num_threads(20)

    opt = TestStudentOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.amp = True if opt.gpu_ids else False
    # opt.epoch='best'

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

    model_s: ColorizationStudentModel = create_model(opt)
    model_s.setup(opt)
    opt.model = 'colorization'
    model_t: ColorizationModel = create_model(opt)
    model_t.setup(opt)
    model_s.eval()
    model_t.eval()
    for param in model_s.netG_student.parameters():
        param.requires_grad = False
    for param in model_t.netG.parameters():
        param.requires_grad = False

    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    webpage = html.HTML(
        web_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}'
    )

    dataset = create_dataset(opt)
    scores = []
    netG_student_time = 0
    netG_time = 0
    for i, data in enumerate(dataset):
        print('processing (%04d)-th image... %s' % (i, data['A_paths']))

        model_s.set_input(data)
        model_t.set_input(data)
        with amp.autocast(enabled=opt.amp):
            model_s.test()
            model_t.test()

        netG_student_time = (netG_student_time * i + model_s.netG_student_time) / (
            i + 1
        )
        netG_time = (netG_time * i + model_t.netG_time) / (i + 1)

        visuals_s = model_s.get_current_visuals()
        visuals_t = model_t.get_current_visuals()
        img_path = model_s.get_image_paths()
        scores.append(model_s.compute_scores())
        save_images_st(
            webpage,
            visuals_s,
            visuals_t,
            img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize,
        )
    webpage.save()
    print('Histogram Intersection: %.4f' % np.mean(scores))
    print('netG_student_time', netG_student_time)
    print('netG_time', netG_time)
