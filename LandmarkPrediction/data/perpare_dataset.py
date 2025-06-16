#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @File    : perpare_dataset.py
# @Time    : 2021/6/2 16:16
# @Author  : John
# @Software: PyCharm
import os
import shutil
import argparse
from random import sample


def filter(paths, k, target_dir):
    for root, dirs, _ in os.walk(paths):
        for dire in dirs:
            if len(os.listdir(os.path.join(root, dire))) < k:
                shutil.move(os.path.join(root, dire), target_dir)


def split_train_test_val(source_dir):
    diretory = os.listdir(source_dir)
    train = sample(diretory, int(len(diretory) * 0.8))
    diretory = list(set(diretory) - set(train))
    val = sample(diretory, int(len(diretory) * 0.5))
    diretory = list(set(diretory) - set(val))
    test = diretory
    for items in ['train', 'test', 'val']:
        if not os.path.exists(os.path.join(source_dir, items)):
            os.mkdir(os.path.join(source_dir, items))
        for i in eval(items):
            shutil.move(os.path.join(source_dir, i), os.path.join(source_dir, items))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default="filter", type=str, help="filter|split")
    parser.add_argument('paths', type=str)
    parser.add_argument('--target_dir', default="", type=str)
    parser.add_argument('--k', default=3, type=int)
    args = parser.parse_args()
    if args.mode == "filter":
        if not os.path.exists(args.target_dir):
            os.mkdir(args.target_dir)
        filter(args.paths, args.k, args.target_dir)
    elif args.mode == "split":
        split_train_test_val(args.paths)
