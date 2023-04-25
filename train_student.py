import os
import pickle
from collections import OrderedDict
from math import inf

import numpy as np
import torch
from torch.cuda import amp
from tqdm import tqdm

from data import create_dataset
from models import create_model
from models.colorization_model import ColorizationModel
from models.colorization_student_model import ColorizationStudentModel
from options.test_options import TestOptions
from options.train_student_options import TrainStudentOption

if __name__ == '__main__':
    opt = TrainStudentOption().parse()
    opt.num_threads = 20
    opt.batch_size = 2
    opt.no_flip = True
    opt.continue_train = False
    opt.amp = True if opt.gpu_ids else False

    opt_t = TestOptions().parse()
    opt_t.batch_size = opt.batch_size

    torch.set_num_threads(opt.num_threads)

    device = (
        torch.device('cuda:{}'.format(opt.gpu_ids[0]))
        if opt.gpu_ids
        else torch.device('cpu')
    )

    opt.value = (
        torch.tensor(np.load('./doc/ab_constant_filter.npy'))
        .unsqueeze(0)
        .expand(opt.batch_size, -1, -1)
        .float()
        .to(device)
    )
    opt_t.value = opt.value

    with open('./doc/s_t_shapes', 'rb') as f:
        s_t_shapes = pickle.load(f)
        opt.s_shapes = s_t_shapes['s_shapes']
        opt.t_shapes = s_t_shapes['t_shapes']
        opt.n_t = s_t_shapes['n_t']
        opt.unique_t_shapes = s_t_shapes['unique_t_shapes']

    model_t: ColorizationModel = create_model(opt_t)
    model_t.setup(opt_t)
    model_t.eval()
    for param in model_t.netG.parameters():
        param.requires_grad = False
    model_t.isTrain = True

    model_s: ColorizationStudentModel = create_model(opt)
    model_s.setup(opt)

    dataset = create_dataset(opt)

    best_loss = inf
    epochs = 10
    for epoch in range(epochs):
        postfix = OrderedDict(
            best_loss=best_loss,
            loss=0,
            AFD=0,
            L1=0,
            perc=0,
            netG_time=0,
            netG_s_time=0,
        )

        with tqdm(
            dataset,
            desc=f'Epoch {epoch + 1}/{epochs}',
            total=len(dataset) // opt.batch_size,
        ) as pbar:
            for i, data in enumerate(pbar, 1):
                with torch.no_grad():
                    model_t.set_input(data)
                    with amp.autocast(enabled=opt.amp):
                        feat_t, fake_imgs_t = model_t.forward()
                    feat_t = [f.detach() for f in feat_t]
                    fake_imgs_t = [f.detach() for f in fake_imgs_t]

                model_s.set_input(data)
                model_s.optimize_parameters(feat_t, fake_imgs_t)

                postfix['best_loss'] = best_loss
                postfix['loss'] = (
                    postfix['loss'] * (i - 1) + model_s.loss_G.detach().item()
                ) / i
                postfix['AFD'] = (
                    postfix['AFD'] * (i - 1) + model_s.loss_AFD.detach().item()
                ) / i
                postfix['L1'] = (
                    postfix['L1'] * (i - 1) + model_s.loss_L1.detach().item()
                ) / i
                postfix['perc'] = (
                    postfix['perc'] * (i - 1) + model_s.loss_perc.detach().item()
                ) / i
                postfix['netG_time'] = (
                    postfix['netG_time'] * (i - 1) + model_t.netG_time
                ) / i
                postfix['netG_s_time'] = (
                    postfix['netG_s_time'] * (i - 1) + model_s.netG_student_time
                ) / i
                pbar.set_postfix(postfix)

                if postfix['loss'] < best_loss:
                    best_loss = postfix['loss']
                    model_s.save_networks('best')
                if i % (len(dataset) // opt.batch_size // 10) == 0:
                    model_s.save_networks('latest')

        model_s.save_networks(i)

# os.system('shutdown now')
