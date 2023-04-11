from argparse import ArgumentParser
from .base_options import BaseOptions


class TrainStudentOption(BaseOptions):
    def initialize(self, parser: ArgumentParser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--display_id', type=int, default=-1)
        parser.set_defaults(model='colorization_student')
        parser.add_argument('--beta', default=200, type=float)
        parser.add_argument('--qk_dim', default=128, type=int)
        self.isTrain = True
        return parser
