import pickle

import numpy as np
import torch
from torch.cuda import amp

from data import create_dataset
from models.networks import define_G, define_G_student
from options.train_student_options import TrainStudentOption
from util import util


def make_input(input):
    return dict(
        A_l=[A_l.to(device) for A_l in input['A_l']],
        R_l=input['R_l'].to(device),
        R_ab=[R_ab.to(device) for R_ab in input['R_ab']],
        hist=input['hist'].to(device),
    )


opt = TrainStudentOption().parse()
opt.num_threads = 0
opt.batch_size = 2
opt.amp = True if opt.gpu_ids else False

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

net_G = define_G(
    opt.input_nc,
    opt.bias_input_nc,
    opt.value,
    opt.isTrain,
    opt.norm,
    opt.init_type,
    opt.init_gain,
    opt.gpu_ids,
)
net_G_student = define_G_student(
    opt.input_nc,
    opt.bias_input_nc,
    opt.value,
    opt.isTrain,
    opt.norm,
    opt.init_type,
    opt.init_gain,
    opt.gpu_ids,
)
net_G.eval()
net_G_student.eval()
for param in net_G.parameters():
    param.requires_grad = False
for param in net_G_student.parameters():
    param.requires_grad = False

dataset = create_dataset(opt)
for data in dataset:
    data = make_input(data)
    with torch.no_grad():
        with amp.autocast(enabled=opt.amp):
            feat_t, _ = net_G(
                data['A_l'][-1],
                data['R_l'],
                data['R_ab'][0],
                data['hist'],
            )
            feat_s, _ = net_G_student(
                data['A_l'][-1],
                data['R_l'],
                data['R_ab'][0],
                data['hist'],
            )
    s_shapes = [f.shape for f in feat_s]
    t_shapes = [f.shape for f in feat_t]
    n_t, unique_t_shapes = util.unique_shape(t_shapes)
    with open('./doc/s_t_shapes', 'wb') as f:
        pickle.dump(
            dict(
                s_shapes=s_shapes,
                t_shapes=t_shapes,
                n_t=n_t,
                unique_t_shapes=unique_t_shapes,
            ),
            f,
        )
    break


with open('./doc/s_t_shapes', 'rb') as f:
    s_t_shapes = pickle.load(f)
