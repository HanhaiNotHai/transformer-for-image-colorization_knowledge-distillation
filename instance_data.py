import pickle
from data import create_dataset
from options.train_student_options import TrainStudentOption


opt = TrainStudentOption().parse()
opt.num_threads = 0
opt.batch_size = 2
dataset = create_dataset(opt)
for data in dataset:
    with open('doc/instance_data', 'wb') as f:
        pickle.dump(data, f)
    break

with open('doc/instance_data', 'rb') as f:
    data = pickle.load(f)
    print(data)
