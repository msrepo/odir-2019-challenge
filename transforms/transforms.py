import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import Lambda, Compose,ToTensor
import torchvision.transforms.functional as tvf

def nonzero_bounding_box(img:np.ndarray, verbose=False):
    '''
    1. split the image into four quadrants: h_left_split, h_right_split, w_top_split, w_bottom_split
    2. find the last non-zero pixel position for left and top splits
    3. find the first non-zero pixel position for right and bottom splits
    return the index of the above 4 values as bounding box (left,top,right,bottom)
    '''
    h,w,c = img.shape

        
    # split image into four quadrants, use the first channel
    left_half_axis_1d = img[h//2,:w//2,0].tolist()
    top_half_axis_1d = img[:h//2,w//2,0].tolist()

    right_half_axis_1d = img[h//2,w//2:,0].tolist()
    bottom_half_axis_1d = img[h//2:,w//2,0].tolist()

    # find first nonzero pixel positions, if no non-zero pixel positions exist, return lower-bounds and upper-bounds
    try:
        h_left = len(left_half_axis_1d) - left_half_axis_1d[::-1].index(0)
    except ValueError as e:
        # could not find zero in the list
        h_left = 0
    
    try:
        w_top = len(top_half_axis_1d) - top_half_axis_1d[::-1].index(0)
    except ValueError as e:
        w_top = 0

    try:
        h_right = w//2 + right_half_axis_1d.index(0)
    except ValueError as e:
        h_right = h
    
    try:
        w_bottom = h//2 + bottom_half_axis_1d.index(0)
    except ValueError as e:
        w_bottom = w

    if verbose:
        print(f'Image size {img.shape}')
        print(h_left,h_right,w_top,w_bottom)
    return h_left,h_right,w_top,w_bottom

def crop_nonzero(img, verbose=False):
    left, right, top, bottom = nonzero_bounding_box(img,verbose=verbose)
    return img[top:bottom,left:right,:]


def pad_to_largest_square(img:torch.Tensor,verbose=False):
    c,h,w = img.shape
    largest_side = max(img.shape)
    if (largest_side - h) != 0 :
        total_pad = largest_side - h 
        # this is the side where we need to pad
        if total_pad % 2 == 0: 
            #even padding
            top = bottom = total_pad // 2
        else:
            top = total_pad // 2
            bottom = total_pad // 2 + 1
    else:
        top = bottom = 0

    if (largest_side - w )!= 0:
        total_pad = largest_side - w
        # this is the side where we need to pad
        if total_pad % 2 == 0:
            # even padding
            left = right = total_pad // 2
        else:
            # odd padding
            left = total_pad // 2
            right = total_pad // 2 + 1
    else:
        left = right = 0

    required_pad = (left,top,right,bottom)
    padded_img =  tvf.pad(img,required_pad,fill=0,padding_mode='constant') 

    if verbose:
        print('Img shape',img.shape)
        print('padding', required_pad)
    return padded_img

img_transform = Compose([
    Lambda(plt.imread),
    Lambda(crop_nonzero),
    ToTensor(),
    Lambda(pad_to_largest_square),
])