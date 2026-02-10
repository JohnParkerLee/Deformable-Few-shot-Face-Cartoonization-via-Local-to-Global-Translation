CUDA_VISIBLE_DEVICES=5 python train_dis2.py --batch 3 --ckpt ./models/source_ffhq.pt --data_path ./processed_data/fernand/ --exp fernand_dis_pg --n_train 10 --iter 10002 --img_freq 200 --save_freq 1000 --size 256 --feat_const_batch 6 --subspace_freq 4

