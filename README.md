## Deformable-Few-shot-Face-Cartoonization-via-Local-to-Global-Translation
Deformable Few-shot Face Cartoonization via Local to Global Translation

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
