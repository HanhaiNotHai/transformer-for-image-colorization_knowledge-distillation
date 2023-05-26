(python -u s_t_shapes.py --gpu_id 0 --preprocess scale_width_and_crop && \
python -u train_student.py --gpu_id 0 --preprocess scale_width_and_crop) >train.log 2>&1
