## Deformable-Few-shot-Face-Cartoonization-via-Local-to-Global-Translation
Deformable Few-shot Face Cartoonization via Local to Global Translation


- [Deformable-Few-shot-Face-Cartoonization-via-Local-to-Global-Translation](#deformable-few-shot-face-cartoonization-via-local-to-global-translation)
  - [Requirements](#requirements)
  - [Stage I, Local Component Translation](#stage-i-local-component-translation)
    - [1. Dataset](#1-dataset)
    - [2. Data preprocss](#2-data-preprocss)
    - [3. Network Training](#3-network-training)
    - [4. Network Testing](#4-network-testing)


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