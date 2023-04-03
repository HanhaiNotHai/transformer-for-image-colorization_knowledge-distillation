import os

import numpy as np

from data import create_dataset
from models import create_model
from options.test_student_options import TestStudentOptions
from util import html
from util.visualizer import save_images

if __name__ == '__main__':
    opt = TestStudentOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    web_dir = os.path.join(opt.results_dir, opt.name,
                           f'{opt.phase}_{opt.epoch}')
    webpage = html.HTML(web_dir,
                        f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}')
    scores = []
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        metrics = model.compute_scores()
        scores.extend(metrics)
        print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path,
                    aspect_ratio=opt.aspect_ratio,
                    width=opt.display_winsize)
    webpage.save()
    print('Histogram Intersection: %.4f' % np.mean(scores))
