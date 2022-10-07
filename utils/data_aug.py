import random
import torch
import numpy as np


def shear(input_skeleton):
    extend = 0.1
    tMatrix = torch.tensor([
        [1, random.uniform(-extend, extend), random.uniform(-extend, extend)],
        [random.uniform(-extend, extend), 1, random.uniform(-extend, extend)],
        [random.uniform(-extend, extend), random.uniform(-extend, extend), 1]
    ])
    output_skeleton = torch.matmul(input_skeleton,tMatrix)
    return output_skeleton


def joint_mask(input_skeleton):
    masknum = 2 #number of joints to mask
    jointnum = input_skeleton.size(dim=1)

    output_skeleton = input_skeleton.clone()

    for i in range(0,masknum):
        joint = np.floor(jointnum * random.random())
        output_skeleton[:,i,:] = 0

    return output_skeleton


def gaussian_noise(input_skeleton):
    f,j,d = input_skeleton.size()
    magnitude = 0.5
    noise = magnitude * torch.normal(0,1,size=(f,j,d))
    output_skeleton = input_skeleton + noise

    return output_skeleton


def flip(input_skeleton):
    '''flip in left-right direction'''
    output_skeleton = input_skeleton.clone()
    output_skeleton[:,:,0] = (-1)*output_skeleton[:,:,0]

    return output_skeleton

def argmentation(skeleton):
    ch = np.floor(4 * random.random())
    if ch == 0:
        output = shear(skeleton)
    elif ch == 1:
        output = joint_mask(skeleton)
    elif ch == 2:
        output = gaussian_noise(skeleton)
    elif ch == 3:
        output = flip(skeleton)

    return output