#!/usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
import pcl
from ipdb import set_trace as pdb

hsv_range = {  # attention! just for uint8 image, not for float32 image
    'blue':   [[115,150,150],[125,255,255]],  # 120,255,255
    'green':  [[55 ,150,150],[65 ,255,255]],  # 60,255,255
    'red':    [[0  ,150,150],[10 ,255,255]],  # 0, 255,255
    'yellow': [[25 ,150,150],[35 ,255,255]],  # 30,255,255
    'purple': [[145,150,150],[155,255,255]],  # 150,255,255
}
color_rgba = {
    'red':   [3, 0, 0, 1],  # 0, 255,255
    'green': [0, 3, 0, 1],  # 60,255,255
    'blue':  [0, 0, 3, 1],  # 120,255,255
    'yellow':[3, 3, 0, 1],  # 30,255,255
    'purple':[3, 0, 3, 1],  # 150,255,255
}

def vis_rgbd_img(img, color=None):
    """
    img: 4 x h x w
    """
    color_img = img[:3,:,:]
    depth_img = img[3,:,:]
    vis_color_img(color_img)
    vis_depth_img(depth_img)
    if color is not None:
        mask = get_mask(color_img, color)
        vis_mask(mask)

def save_rgbd_img(img, color=None, path='../../exp/',):
    color_img = img[:3,:,:]
    depth_img = img[3,:,:]
    save_color_img(color_img, path)
    save_depth_img(depth_img, path)
    if color is not None:
        mask = get_mask(color_img, color)
        save_mask(mask, path)

def vis_color_img(img):
    """
    img: 3 x h x w
    """
    img = img.transpose((1,2,0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    cv2.imshow('color', img)
    cv2.waitKey(0)

def save_color_img(img, path):
    img = img.transpose((1,2,0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 0)
    cv2.imwrite(path + '/color.png', img)

def vis_depth_img(img):
    """
    img: h x w or 1 x h x w
    """
    img = img.squeeze()
    img = cv2.flip(img, 0)
    cv2.imshow('depth', img)
    cv2.waitKey(0)

def save_depth_img(img, path):
    """
    img: h x w or 1 x h x w
    """
    img = img * 100
    img = img.squeeze()
    img = cv2.flip(img, 0)
    cv2.imwrite(path + '/depth.png', img)

def get_mask(img, color):
    """
    img: color img 3 x h x w, rgb
    color: 'yellow',...
    """
    lower, upper = np.array(hsv_range[color])

    img = img.astype('uint8')
    img = img.transpose((1,2,0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = cv2.flip(img, 0)
    mask = cv2.inRange(img, lower, upper)
    return mask

def vis_mask(mask):
    """
    mask: h x w
    """
    cv2.imshow('mask', mask)
    cv2.waitKey(0)

def save_mask(img, path):
    """
    img: h x w or 1 x h x w
    """
    cv2.imwrite(path + '/mask.png', img)

def get_pcd(rgbd_img, cam_mat, cam_pos, cam_f, color):
    """
    rgbd_img: 4 x h x w
    cam_mat: camera rotation matrix, 3x3 np.array 
    cam_pos: camera translation vector, (3,) np.array
    """
    color_img, depth_img = rgbd_img[:3], rgbd_img[3]
    h, w = depth_img.shape
    mask = get_mask(color_img, color)

    x_pix = np.arange(w) - (w-1)/2
    x_pix = np.tile(x_pix, (h,1))
    y_pix = np.arange(h)[:, np.newaxis] - (h-1)/2
    y_pix = np.tile(y_pix, (1,w))
    
    x_pcd = x_pix * depth_img / cam_f
    y_pcd = y_pix * depth_img / cam_f

    mask_index = mask > 0

    x_pcd = x_pcd[mask_index]
    y_pcd = -y_pcd[mask_index]
    z_pcd = -depth_img[mask_index]
    pcd = np.stack([x_pcd, y_pcd, z_pcd], axis=0)

    pcd = cam_mat.dot(pcd) + cam_pos.reshape(3,1)
    pcd = pcd.T  # n x 3

    return pcd

def save_pcd(pcd, path='../../exp/'):
    rgb = np.ones((pcd.shape[0], 1)) * 255
    pcd_rgb = np.concatenate((pcd, rgb), axis=1).astype('float32')
    cloud = pcl.PointCloud_PointXYZRGB(pcd_rgb)
    pcl.save(cloud, path + 'cloud.pcd')
