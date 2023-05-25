import os
import pickle
from collections import OrderedDict
from math import inf

import numpy as np
import torch
from torch.cuda import amp
from tqdm import tqdm, trange

from data import create_dataset
from models import create_model
from models.colorization_model import ColorizationModel
from models.colorization_student_model import ColorizationStudentModel
from options.test_options import TestOptions
from options.train_student_options import TrainStudentOption


def main():
    opt = TrainStudentOption().parse()
    ####################
    opt.num_threads = 20
    ##################
    opt.batch_size = 2
    opt.no_flip = True
    #########################
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
    if opt.gpu_ids:
        model_t.netG.module.isTrain = True
    else:
        model_t.netG.isTrain = True

    model_s: ColorizationStudentModel = create_model(opt)
    #################
    # opt.epoch = '500'
    model_s.setup(opt)

    dataset = create_dataset(opt)

    ############
    epochs = 100
    for epoch in trange(1, epochs + 1):
        losses = OrderedDict(
            G=[0] * 10,
            AFD=[0] * 10,
            mse=[0] * 10,
            L1=[0] * 10,
            perc=[0] * 10,
            hist=[0] * 10,
            sparse=[0] * 10,
        )
        postfix = OrderedDict(
            G=0,
            AFD=0,
            mse=0,
            L1=0,
            perc=0,
            hist=0,
            sparse=0,
        )

        with tqdm(
            dataset,
            desc=f'Epoch {epoch}/{epochs}',
            total=len(dataset) // opt.batch_size,
        ) as pbar:
            for data in pbar:
                with torch.no_grad():
                    model_t.set_input(data)
                    with amp.autocast(enabled=opt.amp):
                        feat_t_AFD, feat_t_mse, fake_imgs_t = model_t.forward()
                    feat_t_AFD = [f.detach() for f in feat_t_AFD]
                    feat_t_mse = [f.detach() for f in feat_t_mse]
                    fake_imgs_t = [f.detach() for f in fake_imgs_t]

                model_s.set_input(data)
                model_s.optimize_parameters(feat_t_AFD, feat_t_mse, fake_imgs_t)

                for k, v in model_s.get_current_losses().items():
                    losses[k] = losses[k][1:] + [v]
                    postfix[k] = sum(losses[k]) / 10
                pbar.set_postfix(postfix)

        model_s.save_networks('latest')

        if epoch % (epochs // 10) == 0:
            model_s.save_networks(
                f'{opt.epoch}_{epoch}' if opt.continue_train else f'{epoch}'
            )

    ################################
    # os.system('/usr/bin/shutdown')


if __name__ == '__main__':
    main()
