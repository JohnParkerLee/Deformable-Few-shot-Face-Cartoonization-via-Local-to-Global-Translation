## Deformable-Few-shot-Face-Cartoonization-via-Local-to-Global-Translation
Deformable Few-shot Face Cartoonization via Local to Global Translation

**Code are still preparing**


- [Deformable-Few-shot-Face-Cartoonization-via-Local-to-Global-Translation](#deformable-few-shot-face-cartoonization-via-local-to-global-translation)
  - [Requirements](#requirements)
  - [Stage I, Local Component Translation](#stage-i-local-component-translation)
    - [1. Dataset](#1-dataset)
    - [2. Data preprocss](#2-data-preprocss)
    - [3. Network Training](#3-network-training)
    - [4. Network Testing](#4-network-testing)
  - [Stage I, Landmark Prediction](#stage-i-landmark-prediction)
    - [1. Data preprocess](#1-data-preprocess)
    - [2. Data augmentation](#2-data-augmentation)
    - [3. Network Training](#3-network-training-1)
      - [Training](#training)
      - [Finetuning for new style](#finetuning-for-new-style)
    - [4. Pre-trained models](#4-pre-trained-models)
      - [~~Inference for visulization~~](#inference-for-visulization)
  - [Stage II, LocalComponentNetwork](#stage-ii-localcomponentnetwork)
    - [Train](#train)
    - [Inference](#inference)
  - [Stage III, Global Refinement](#stage-iii-global-refinement)
    - [1. Training Process](#1-training-process)
    - [2. pre-trained global refinement network](#2-pre-trained-global-refinement-network)
    - [3. pre-trained psp models](#3-pre-trained-psp-models)
    - [4. pre-trained fine-tuned models](#4-pre-trained-fine-tuned-models)


### Requirements

- python==3.7.13
- pytorch==1.7.1
- opencv-python==4.5.5.64
- matplotlib==3.5.2
- tensorboard==2.9.0

### Stage I, Local Component Translation 

#### 1. Dataset

​	[CFD—Chicago Face Database](https://www.chicagofaces.org/)

​	Use `extract_image.py` to extract all the images and divide them into training and te st sets.

#### 2. Data preprocss

​	We have provided the [processed style data](https://drive.google.com/drive/folders/1KV2oxNO-rd61qxonzlcC11tNL0NzhCDF?usp=sharing).

- Landmark detection(5 points)

  **For real face**, use `alignemnt_script\data_process` to obtain facial landmark(5 points).

  ```shell
  # Get the 5 points landmarks
  python data_process.py --mode detect_mtcnn --data_path /path/to/img --save_path /path/to/save
  ```

  If you need to adjust the detected landmark, you can run the following script. Don't forget to add a landmark path to the `transToJson` function  .

  ```shell
  # Convert .txt to .json to adjust landmarks
  python data_process.py --mode trans_to_json --data_path
  /path/to/img --save_path /path/to/save
  ```

  And use [labelme](https://github.com/wkentaro/labelme) to mark landmarks.

  **For the cartoon image**, you can manually mark landmarks using [labelme](https://github.com/wkentaro/labelme), and then use the scripts to convert the `.json` file to the `.mat` file.  

  ```shell
  # Convert .json to .mat for alignment
  python data_process.py --mode json_to_mat --data_path /path/to/json --save_path /path/to/save
  ```

- Face alignment

  Next, following the [APDrawingGAN](https://github.com/yiranran/APDrawingGAN/tree/master/preprocess) instructions to align, resize and crop images to 512x512 and prepare facial landmarks (5 points). **Place the image in the corresponding folder. For example, real face images should be placed in `trainA_img`, and their 5 points landmarks should be placed in `/feat5/trainA_img`.**
  
- Landmark detection (17 points)

  To obtain 17 points landmarks, use [face-of-the-art](https://faculty.runi.ac.il/arik/site/foa/face-of-art.asp) and save the landmarks to their corresponding folder **/datasets/feat17/···**. 

  Alternatively, you can use [labelme](https://github.com/wkentaro/labelme) to adjust the landmarks, and then use the following script to convert them to the `.txt` format.

  ```shell
  # Convert .json to .txt
  python data_process.py --mode json_to_txt --data_path /path/to/json --save_path /path/to/save
  ```

- File tree

  /datasets

  -- /trainA_img

  -- /trainB_sketches

  -- /trainB_Amedeo

  -- /testA_img

  -- ...

  -- /feat5

  --   -- /trainA_img

  --   -- /trainB_sketches

  --   -- /trainB_Amedeo

  --   -- /testA_img

  -- ···

  -- /feat17

  --   -- /trainA_img

  --   -- /trainB_sketches

  --   -- /trainB_Amedeo

  --   -- /testA_img

  --   -- ···


#### 3. Network Training

Training scripts

```shell
python train_pair.py --tensorboard <tensorboard_name> --dataroot ./path/to/data/root --name <project_name> --model wocat_unpair_ori_cycle_gan --netG cyclegan --netD m_dis --init_type xavier --direction AtoB --dataset_mode unpairconsistency --norm instance --gpu_ids 0 --display_port 8103 --batch_size 5 --preprocess none --n_epochs 20 --n_epochs_decay 15 --ngf 64 --save_epoch_freq 5 --num_threads 4 --no_dropout --input_nc 3 --output_nc 3 --component <eyer/nose/mouth> --lambda_sw 5e-6 --lambda_rel 0.0 --lambda_contextual 0.0 --lambda_hedgan 0.0 --lambda_avg 100.0 --lambda_identity 0.1 --lambda_trunc 0.0 --crop_size 256 --dataset _<dataset_name>
```
#### 4. Network Testing

The provided pre-trained local component translation models.

|    Styles     |                            Models                            |
| :-----------: | :----------------------------------------------------------: |
| Minivision-ai | [minivision](https://drive.google.com/drive/folders/1RGsOJovz22k7yYLxt7uLCtrD6NC59Ef6?usp=sharing) |
|   sketches    | [sketches](https://drive.google.com/drive/folders/1iAsLRLfvKVbl5-YSX1-ZUEOFSVubaxXz?usp=sharing) |
|    Amedeo     | [Amedeo](https://drive.google.com/drive/folders/1TApf9MH4PQ6EGcfCa1H_kJA3PeJXRrH-?usp=sharing) |

Test scripts

```shell
python test.py --dataroot ./path/to/data/root --name <project_name> --model wocat_unpair_ori_cycle_gan --netG cyclegan --netD m_dis --init_type xavier --direction AtoB --dataset_mode testconsistency --norm instance --gpu_ids 1 --batch_size 1 --preprocess none --ngf 64 --num_threads 2 --no_flip --no_dropout --input_nc 3 --output_nc 3 --component <eyer/nose/mouth>
```



### Stage I, Landmark Prediction

#### 1. Data preprocess

Randomly select 100 images from the training dataset and use the landmarks detected by [face-of-art](https://faculty.runi.ac.il/arik/site/foa/face-of-art.asp) as training data for landmark prediction. Remeber to convert `.txt` to `.json`.

#### 2. Data augmentation

Use `alignment_script\data_process.py` to augment the landmarks.

```shell
python data_process.py --mode augmentation --data_path
/path/to/landmark_68-points_json --save_path /path/to/save
```

#### 3. Network Training

The network directory is called `LandmarkPrediction`.

##### Training

Training scripts for 5 points

```shell
python train.py --dataroot /path/to/data/root --style_data_paths path/to/augmentation/json/path --content_data_paths path/to/landmark_68-points_json --dataset_mode final --load_size 512 --k 2 --model landmark_gan --netG landmarknet --lambda_GAN 0.0 --input_nc 10 --output_nc  10 --lambda_L1 100 --gpu_ids 0 --batch_size 16 --triplet 80 --num_threads 12 --norm none --name 5_points_trian --display_freq 10 --niter 500 --niter_decay 300 --phase train --validate_freq 100 --part_class points_5 --init_type normal
```

Training scripts for 17 points

```shell
python train.py --dataroot /path/to/data/root --style_data_paths path/to/augmentation/json/path --content_data_paths path/to/landmark_68-points_json --dataset_mode final --load_size 512 --k 2 --model landmark_gan --netG landmarknet --lambda_GAN 0.0 --input_nc 34 --output_nc  34 --lambda_L1 100 --gpu_ids 0 --batch_size 16 --triplet 80 --num_threads 12 --norm none --name 17_points_train --display_freq 10 --niter 800 --niter_decay 600 --phase train --validate_freq 100 --part_class head --init_type normal
```

##### Finetuning for new style

Finetune scripts for 5-point landmarks

```shell
python train.py --dataroot /path/to/data/root --style_data_paths path/to/style/path --content_data_paths path/to/landmark_68-points_json --dataset_mode finetune --load_size 512 --k 2 --model landmark_gan --netG landmarknet --lambda_GAN 0.0 --input_nc 10 --output_nc  10 --lambda_L1 100 --gpu_ids 0 --batch_size 16 --triplet 60 --num_threads 12 --norm none --name 5_points_finetune --display_freq 10 --niter 800 --niter_decay 200 --phase train --validate_freq 100 --part_class points_5 --continue --init_type normal --epoch 800 --epoch_count 801 --lr 0.00002 --finetune True
```

Finetune scripts for 17-point landmark

```shell
python train.py --dataroot /path/to/data/root --style_data_paths path/to/style/path --content_data_paths path/to/landmark_68-points_json --dataset_mode finetune --load_size 512 --k 2 --model landmark_gan --netG landmarknet --lambda_GAN 0.0 --input_nc 10 --output_nc  10 --lambda_L1 100 --gpu_ids 0 --batch_size 16 --triplet 60 --num_threads 12 --norm none --name 17_points_finetune --display_freq 10 --niter 1400 --niter_decay 400 --phase train --validate_freq 100 --part_class head --continue --init_type normal --epoch 1400 --epoch_count 1401 --lr 0.00002 --finetune True
```

#### 4. Pre-trained models

The pre-trained models with 5 points and 17 points, as well as the models fine-tuned on the Amedeo style. [link](https://drive.google.com/drive/folders/1cdHJg4eqB7_EkHYDpF46_--e5EZZB2Ob?usp=sharing)


##### ~~Inference for visulization~~

Inference for 5 points

```shell
python test.py --dataroot /path/to/data/root --style_data_paths path/to/style/path --content_data_paths path/to/landmark_68-points_json --results_dir path/to/save --dataset_mode final_test --phase test --num_test 100 --k 10 --input_nc 10 --output_nc 10 --gpu_ids 0 --model landmark_gan --netG landmarknet --name 5_points_finetune --load_size 512 --crop_size 512 --eval --norm none --part_class points_5 --epoch 1000
```

Inference for 17 points

```shell
python test.py --dataroot /path/to/data/root --style_data_paths path/to/style/path --content_data_paths path/to/landmark_68-points_json --results_dir path/to/save --dataset_mode final_test --phase test --num_test 100 --k 10 --input_nc 34 --output_nc 34 --gpu_ids 0 --model landmark_gan --netG landmarknet --name 17_points_finetune --load_size 512 --crop_size 512 --eval --norm none --part_class head --epoch 1800
```

### Stage II, LocalComponentNetwork

#### Train


#### Inference

Navigate to the `cyclegan` directory and run the following script to obtain `intermediate results`, if you need landmark prediction:

```shell
python test.py --dataroot /path/to/data/root --name <project_name> --model combine --netG cyclegan --netD m_dis --init_type xavier --direction AtoB --dataset_mode combine --norm instance --gpu_ids 0 --batch_size 1 --preprocess none --ngf 64 --num_threads 2 --no_flip --no_dropout --input_nc 3 --output_nc 3 --num_test 1500 --pretrain_G_content_5 /path/to/landmark_prediction_network/checkpoints/5_points_finetune/1000_net_G_content.pth --pretrain_G_style_5 /path/to/landmark_prediction_network/checkpoints/5_points_finetune/1000_net_G_style.pth --pretrain_G_content_17 /path/to/landmark_prediction_network/checkpoints/17_points_finetune/1800_net_G_content.pth --pretrain_G_style_17 /path/to/landmark_prediction_network/checkpoints/17_points_finetune/1800_net_G_style.pth
--need_landmark_predict
```

If you don't need landmark prediction, use the following script:

```shell
python test.py --dataroot /path/to/data/root --name <project_name> --model combine --netG cyclegan --netD m_dis --init_type xavier --direction AtoB --dataset_mode combine --norm instance --gpu_ids 0 --batch_size 1 --preprocess none --ngf 64 --num_threads 2 --no_flip --no_dropout --input_nc 3 --output_nc 3 --num_test 1500
```

### Stage III, Global Refinement

#### 1. Training Process

1. Prepare the data and StyleGAN2 checkpoint according to the [GAN adaptation](https://github.com/utkarshojha/few-shot-gan-adaptation) instructions.

2. Assign the local component network checkpoint path as an argument.

3. Train the `Global Refinement Network`

   ```shell
   CUDA_VISIBLE_DEVICES=0 python train_dis2.py --batch 3 --ckpt ./models/source_ffhq.pt --data_path ./processed_data/<dataset>/ --exp <project_name> --n_train 10 --iter 2002 --img_freq 200 --save_freq 1000 --size 256 --feat_const_batch 6 --subspace_freq 2
   --cos_wt 1.0 --center_wt 125
   --ckpt_netG_eye /path/to/local/eyer/checkpoint --ckpt_netG_nose /path/to/local/nose/checkpoint --ckpt_netG_mouth /path/to/local/mouth/checkpoint
   ```

   Or you can do not assign the local component network checkpoint path:

   ```shell
   CUDA_VISIBLE_DEVICES=0 python train.py --batch 3 --ckpt ./models/source_ffhq.pt --data_path ./processed_data/<dataset>/ --exp <project_name> --n_train 10 --iter 2002 --img_freq 200 --save_freq 1000 --size 256 --feat_const_batch 6 --subspace_freq 6 --cos_wt 1.0 --center_wt 125
   ```

   After training, use the Global Refinement Network to generate 10,000 images for training GAN inversion.

   ```shell
   CUDA_VISIBLE_DEVICES=0 python generate.py --ckpt_target /path/to/model/ --n_sample 10000
   ```

4. Train the [PSP network](https://github.com/eladrich/pixel2style2pixel) using the `source_ffhq` as the StyleGAN checkpoint.


5. inference

   Assign the related path in `inference_fine_tune.py` and run the following scripts:

   ```shell
   CUDA_VISIBLE_DEVICES=0 python inference_fine_tune.py
   ```




#### 2. pre-trained global refinement network

|                            model                             |
| :----------------------------------------------------------: |
| [sketches](https://drive.google.com/drive/folders/1WMsEPtKy4G0zYLC8JCHO4i-pc6OFwI22?usp=sharing) |
| [Amedeo](https://drive.google.com/drive/folders/1aaZKjqs-WcwcWbW-PuA1bYQHDgyl4SNM?usp=sharing) |
| [minivision](https://drive.google.com/drive/folders/1R3NLmpLoQAGsBVRTc6BVdSqEj9Uue_AI?usp=sharing) |



#### 3. pre-trained psp models

|                            models                            |
| :----------------------------------------------------------: |
| [ffhq-to-ffhq](https://drive.google.com/file/d/1_2M8zJ2YivKRAHlDoh_e7jt1YSk5YibD/view?usp=sharing) |
| [sketches-to-sketches](https://drive.google.com/file/d/1mtYSBvlw7APBr_CKm0KYHkN1qp3rMd08/view?usp=sharing) |
| [Amedeo-to-Amedeo](https://drive.google.com/file/d/1BmjQO3Rt4UD1zRMyO7jBcAQ2cea9-3hB/view?usp=sharing) |
| [minivision-to-minivision](https://drive.google.com/file/d/1a7rutODHeoQXbqYPCBmFU9h8C-ndseUg/view?usp=sharing) |



#### 4. pre-trained fine-tuned models

|                            model                             |
| :----------------------------------------------------------: |
| [sketches](https://drive.google.com/file/d/1VAgOVyJ1NSY1kiNocWLj3TPfgesEMK7P/view?usp=sharing) |
| [Amedeo](https://drive.google.com/file/d/1zQfFc1Xm2Z928UGUylGbiTzuTEx8B2mu/view?usp=sharing) |
| [minivision](https://drive.google.com/file/d/1gavA4JuAFwDve3w1NtgvRwHKr78wkklH/view?usp=sharing) |