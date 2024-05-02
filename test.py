import numpy as np
import cv2
from PIL import Image
import argparse
import torch
import os
from stable_diffusion import *
from depth2normal import *


parser = argparse.ArgumentParser(description ='add input file')
parser.add_argument("-i",type=str, help="input file")
parser.add_argument("-m", type=int, help="0 for txt2img, 1 for txt-depth2img, 2 for txt-normal2img, 3 for txt-depth-normal2img")
parser.add_argument("-p", type=str, help="txt prompt")
args = parser.parse_args()



if __name__ == '__main__':
    print("CUDA available: ", torch.cuda.is_available())

    input_file_path = args.i
    name = None
    mode = args.m
    prompt = args.p
    depth_img = None
    if mode != 0:
        name = os.path.basename(input_file_path).split('.')[0]
        ext = os.path.basename(input_file_path).split('.')[-1]

        if ext == 'png':
            depth_img = cv2.imread(input_file_path, -1)
            # depth_img = cv2.resize(depth_img, (512, 512))
        elif ext == 'npy':
            depth_img = np.load(input_file_path)
            depth_img = depth_img[:, :, None]
            depth_img = np.concatenate([depth_img, depth_img, depth_img], axis=2).astype(np.uint8)


    if mode == 0:
        save_path = os.path.join('output', f'{mode}_txt2img_out.png')
        txt2img(prompt, save_path)
    elif mode == 1:
        depth_img = Image.fromarray(depth_img)
        depth_img.save("7_depth.png")
        save_path = os.path.join('output', name + '_txt_depth2img_out.png')
        txt_depth2img(depth_img, prompt, save_path)
    elif mode == 2:
        normal_img = d2n(depth_img)
        # normal_img = cv2.imread('bird_normal.png', -1)
        normal_img = Image.fromarray(normal_img)
        normal_img.save(os.path.join('output', name+'_normal.png'))
        save_path = os.path.join('output', name + '_txt_normal2img_out.png')
        txt_normal2img(normal_img, prompt, save_path)
    elif mode == 3:
        normal_img = d2n(depth_img)
        depth_img = Image.fromarray(depth_img)
        normal_img = Image.fromarray(normal_img)
        save_path = os.path.join('output', name + '_txt_depth_normal2img_out.png')
        txt_depth_normal2img(depth_img, normal_img, prompt, save_path)
    else:
        print("Bad argument!! Try again!!")
