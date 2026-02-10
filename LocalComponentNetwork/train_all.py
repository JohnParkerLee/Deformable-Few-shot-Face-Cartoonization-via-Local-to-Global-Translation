# set -ex
# python train.py  --dataroot ./datasets/cyclenew --name mouth --model wocat_unpair_ori_cycle_gan --netG cyclegan --netD m_dis --init_type xavier --direction AtoB --dataset_mode unpairconsistency --norm instance --gpu_ids 0 --display_port 8103 --batch_size 5 --preprocess none --n_epochs 25 --n_epochs_decay 20 --ngf 64 --save_epoch_freq 5 --num_threads 4 --no_flip --no_dropout --input_nc 3 --output_nc 3
import os
component_list = ['eyer']
model = {'unpair': 'wocat_unpair_ori_cycle_gan'}
sw = {'with_sw': '1e-5', 'wo_sw':'0.0'}
rel = {'wo_rel': '0.0'}
context = {'wo_context': '0.0'}
hed = {'wo_hed': '0.0'}
avg = {'with_avg':'100', 'wo_avg':'0.0'}
dataset = ['_zijie','_apd','_sketches','manga','_dacheng''_dacheng2']
default_string = 'python3.6 -u pytorch-CycleGAN-and-pix2pix/train.py --dataroot ./datasets/cyclenew --netG cyclegan --netD m_dis --init_type xavier --direction AtoB --dataset_mode unpairconsistency --norm instance --display_port 8103 --batch_size 5 --preprocess none --n_epochs 25 --n_epochs_decay 20 --ngf 64 --save_epoch_freq 5 --num_threads 4 --no_flip --no_dropout --input_nc 3 --output_nc 3'
for component in component_list:
   for modelk, modelv in model.items():
       for swk, swv in sw.items():
           for relk, relv in rel.items():
               for contextk, contextv in context.items():
                 for hedk, hedv in hed.items():
                     for avgk, avgv in avg.items():
                       name = component + '_' + modelk + '_' + swk + '_' + relk + '_' + contextk + '_' + hedk + '_' + avgk + '_' + 'siggraph_'+dataset[0]+'_ablation'

                       command = default_string + ' --tensorboard ' + name + ' --name ' + name + ' --model ' + modelv + ' --component ' + component + ' --lambda_sw ' + swv + ' --lambda_rel ' + relv+ ' --lambda_contextual ' + contextv + ' --lambda_hedgan ' + hedv + ' --lambda_avg ' + avgv + ' --gpu_ids 0 ' + ' --dataset ' +dataset[0]
                       os.system(command)


