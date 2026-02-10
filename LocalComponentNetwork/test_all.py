# set -ex
# python train.py  --dataroot ./datasets/cyclenew --name mouth_unpair_wrel --model unpair_ori_cycle_gan --netG cyclegan --netD m_dis --init_type xavier --direction AtoB --dataset_mode consistency --norm instance --gpu_ids 0 --display_port 8103 --batch_size 5 --preprocess none --n_epochs 25 --n_epochs_decay 20 --ngf 64 --save_epoch_freq 5 --num_threads 4 --no_flip --no_dropout --input_nc 3 --output_nc 3
import os
component_list = ['eyer']
model = {'unpair': 'wocat_unpair_ori_cycle_gan'}
sw = {'with_sw': '1e-5','wo_sw':'0.0'}
rel = {'wo_rel': '0.0'}
context = {'wo_context': '0.0'}
hed = {'wo_hed': '0.0'}
avg = {'with_avg':'100','wo_avg':'0.0'}
dataset = ['_zijie', '_apd','_sketches','_dacheng', '_manga']
default_string = 'python3.6 -u pytorch-CycleGAN-and-pix2pix/test.py  --dataroot ./datasets/cyclenew --netG cyclegan --netD m_dis --init_type xavier --direction AtoB --dataset_mode testconsistency --norm instance --gpu_ids 0 --batch_size 1 --preprocess none --ngf 64 --num_threads 4 --no_flip --no_dropout --input_nc 3 --output_nc 3'
for i in range(len(dataset)):
    for component in component_list:
        for modelk, modelv in model.items():
            for swk, swv in sw.items():
                for relk, relv in rel.items():
                    for contextk, contextv in context.items():
                      for hedk, hedv in hed.items():
                          for avgk, avgv in avg.items():
                            name = component + '_' + modelk + '_' + swk + '_' + relk + '_' + contextk + '_' + hedk + '_' + avgk + '_' + 'siggraph_'+dataset[i]+'_ablation'

                            command = default_string + ' --name ' + name + ' --model ' + modelv + ' --component ' + component+ ' --dataset ' +dataset[i]
                            os.system(command)




