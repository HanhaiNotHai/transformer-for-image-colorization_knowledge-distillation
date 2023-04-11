from math import inf
import pickle
import torch
from torch import Tensor
from tqdm import tqdm

from data import create_dataset
from models import create_model
from models.colorization_model import ColorizationModel
from models.colorization_student_model import ColorizationStudentModel
from models.networks import define_G, define_G_student
from options.test_options import TestOptions
from options.train_student_options import TrainStudentOption
from util import util


def make_input(input: dict[list[Tensor | str] | Tensor]) -> dict[str, list[Tensor | str] | Tensor]:
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
    opt_t = TestOptions().parse()

    opt = TrainStudentOption().parse()
    opt.num_threads = 20
    opt.batch_size = 2
    opt.no_flip = True
    opt.continue_train = False

    device = torch.device('cuda:{}'.format(
        opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    net_G = define_G(
        opt.input_nc, opt.bias_input_nc, opt.output_nc,
        opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids
    )
    net_G_student = define_G_student(
        opt.input_nc, opt.bias_input_nc, opt.output_nc,
        opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids
    )
    with open('doc/instance_data', 'rb') as f:
        data = pickle.load(f)
    data = make_input(data)
    with torch.no_grad():
        feat_t, _ = net_G(
            data['real_A_l'][-1], data['real_R_l'][-1], data['real_R_ab'][0],
            data['hist'], data['ab_constant'], device
        )
        feat_s, _ = net_G_student(
            data['real_A_l'][-1], data['real_R_l'][-1], data['real_R_ab'][0],
            data['hist'], data['ab_constant'], device
        )
    opt.s_shapes = [f.shape for f in feat_s]
    opt.t_shapes = [f.shape for f in feat_t]
    opt.n_t, opt.unique_t_shapes = util.unique_shape(opt.t_shapes)

    model_t: ColorizationModel = create_model(opt_t)
    model_t.setup(opt_t)
    model_t.eval()

    model_s: ColorizationStudentModel = create_model(opt)
    model_s.setup(opt)

    dataset = create_dataset(opt)

    epochs = 10
    best_loss = inf
    for epoch in range(epochs):
        with tqdm(desc=f'Epoch {epoch + 1}/{epochs}', total=len(dataset) // opt.batch_size) as t:
            for i, data in enumerate(dataset):
                data = make_input(data)

                with torch.no_grad():
                    model_t.set_input(data)
                    model_t.forward()
                    feat_t = [f.detach() for f in model_t.feat_t]

                model_s.set_input(data)
                model_s.optimize_parameters(feat_t)

                losses = model_s.get_current_losses()
                loss = sum(losses.values())
                t.set_postfix(
                    loss=list(losses.items()),
                    netG_time=model_t.netG_time,
                    netG_s_time=model_s.netG_student_time,
                )
                t.update(1)

                if i % 1000 == 0:
                    model_s.save_networks('latest')
                    if loss < best_loss:
                        model_s.save_networks('best')
        
        model_s.save_networks('latest')
        if loss < best_loss:
            model_s.save_networks('best')
        model_s.update_learning_rate()
