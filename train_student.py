import pickle

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
        .repeat(opt.batch_size, 1, 1)
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
    model_t.isTrain = True

    model_s: ColorizationStudentModel = create_model(opt)
    model_s.setup(opt)

    dataset = create_dataset(opt)

    epochs = 10
    for epoch in range(epochs):
        with tqdm(
            desc=f'Epoch {epoch + 1}/{epochs}',
            total=len(dataset) // opt.batch_size,
        ) as t:
            for i, data in enumerate(dataset, 1):
                with torch.no_grad():
                    model_t.set_input(data)
                    with amp.autocast(enabled=opt.amp):
                        feat_t, fake_imgs_t = model_t.forward()
                    feat_t = [f.detach() for f in feat_t]
                    fake_imgs_t = [f.detach() for f in fake_imgs_t]

                model_s.set_input(data)
                model_s.optimize_parameters(feat_t, fake_imgs_t)

                t.set_postfix_str(
                    f'loss={model_s.loss_G:.2} '
                    + ' '.join(
                        f'{k}={v:.2e}' for k, v in model_s.get_current_losses().items()
                    )
                    + f' netG_time={model_t.netG_time:.3} netG_s_time={model_s.netG_student_time:.3}'
                )
                t.update(1)

                if i % (len(dataset) // opt.batch_size // 10) == 0:
                    model_s.save_networks('latest')

        model_s.save_networks('latest')
