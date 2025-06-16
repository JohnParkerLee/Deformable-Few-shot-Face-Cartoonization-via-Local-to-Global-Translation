import os
import argparse
import shutil

IMG_EXTENSIONS = ['png', 'jpg', 'jpeg','JPG','PNG']
Basepath = os.path.abspath(os.path.dirname(__file__))
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def extract_img(source_dir, target_dir):
    source_dir = os.path.join(Basepath, source_dir)
    target_dir = os.path.join(Basepath, target_dir)
    print(source_dir, target_dir)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if is_image_file(f):
                path = os.path.join(root, f)
                shutil.copyfile(path, os.path.join(target_dir,f))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = "Extract from CFD dataset")
    parser.add_argument("--source_dir", type = str, help = "source directory")
    parser.add_argument("--target_dir", type = str, help = "target directory")
    args = parser.parse_args()
    extract_img(args.source_dir, args.target_dir)
