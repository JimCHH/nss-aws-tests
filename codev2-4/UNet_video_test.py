#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:07:35 2021

@author: lab70929
"""

from __future__ import print_function
import torch
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet
import parameters
import cv2
import glob

start_time = time.time()
device = torch.device("cuda")

def Unet_test(images, model, Size_X, Size_Y):
    batch_size = len(images)
    t = []
    t1 = time.time()
    inputImg = []
    inputImg_BK = []
    
    for image in images:
        
        image = cv2.resize(image, (Size_X, Size_Y), interpolation=cv2.INTER_CUBIC)
        inputImg_BK.append(image.copy())  
        image = image.astype(np.float32)/255
        image = image[np.newaxis, :]

        inputImg.append(image)
    
    inputImg = np.array(inputImg)
        
    t2 = time.time()
    t.append(t2-t1)
    
    tf_images = torch.from_numpy(inputImg)
    tf_images = tf_images.to(device)
    output = model(tf_images)
    output_bk = output[:, 0].clone().detach().cpu().numpy()
    
    t3 = time.time()
    t.append(t3-t2)
    
    
    for i in range(batch_size):
        temp = output_bk[i]
        gt_temp = inputImg_BK[i]
        ttt = output_bk[i]
        ttt[ttt < 0.5] = 0
        ttt[ttt >= 0.5] = 1
        if np.count_nonzero(ttt) == 0:
            temp[temp < 0.25] = 0
            temp[temp >= 0.25] = 1
        else:
            temp[temp < 0.5] = 0
            temp[temp >= 0.5] = 1
    
        ## Connected Component Analysis
        if np.count_nonzero(temp) != 0:
            _, labels, stats, center = cv2.connectedComponentsWithStats(temp[:, :].astype(np.uint8))
    
            stats = stats[1:, :]
            pupil_candidate = np.argmax(stats[:, 4]) + 1
            temp[:, :][labels != pupil_candidate] = 0
            
        gt_temp[temp == 1] = 255
        output_bk[i] = temp
       
    t4 = time.time()
    t.append(t4-t3)
    
    return inputImg_BK,t







    
name_load_model = 'trained_model/UNet/'
# name_load_model = './trained_model_bk/Cross_validation/base/4f32ch/'
cross_val_num = 18

Size_X = parameters.Size_X
Size_Y = parameters.Size_Y

fourcc = cv2.VideoWriter_fourcc(*'mp4v')


model = UNet(n_channels=1, n_classes=1, bilinear=True)
if os.path.exists(name_load_model):
    load_saved_model_name = parameters.find_latest_model_name(name_load_model, cross_val_num)
    model.load_state_dict(torch.load(load_saved_model_name))
    print(parameters.C_GREEN + 'Check point Successfully Loaded' + parameters.C_END)
else:
    print(parameters.C_RED + 'Check point Not Found' + parameters.C_END)


model.eval()
model.to(device)

#left_roi = [[176,0],[540,300]]
#right_roi = [[750,0],[1130,340]]
left_roi = []
right_roi = []

video_folder = "../Result/"
fps = []
video_list = glob.glob(video_folder+"*/*.mp4")
save_time = []
#for video_name in video_list:
video_name = video_list[0]
output_name = "testing_video.mp4"

print("video_name = ", video_name)

out = cv2.VideoWriter(output_name, fourcc, 210, (384, 144), 0)
#video_name = "video/test/111111_A000000000.avi"
cap = cv2.VideoCapture(video_name)
frame_count = 0


batch_size = 16
counter = 0
images = []
temp_image = []

while (cap.isOpened()):
#    print(counter)

    
    
    if counter<batch_size//2:
        
        ret, frame = cap.read()
        if ret == False:
            break
    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if left_roi ==[]:
            left_eye = frame[:, :640]
            right_eye = frame[:, 640:]
        else:
            left_eye = frame[left_roi[0][1]:left_roi[1][1], left_roi[0][0]:left_roi[1][0]]
            right_eye = frame[right_roi[0][1]:right_roi[1][1], right_roi[0][0]:right_roi[1][0]]
    
        images.append(left_eye)
        images.append(right_eye)
        counter += 1

    else:
        images, times = Unet_test(images, model, Size_X, Size_Y)
        save_time.append(times)
        for i in range(batch_size//2):
            left_result = images[i*2].astype(np.uint8)    
            right_result = images[i*2+1].astype(np.uint8)
            result = np.concatenate((left_result, right_result), axis = 1)
            out.write(result)
            # cv2.imshow('frame', result)
        

        frame_count +=batch_size
        images = []
        counter = 0

    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break       
    
end_time = time.time()
fps.append(frame_count / (end_time - start_time))
cap.release()
out.release()
cv2.destroyAllWindows()





output_video = cv2.VideoCapture(output_name)
length = int(output_video.get(cv2.CAP_PROP_FRAME_COUNT))
print("output frames = " , length )


cap = cv2.VideoCapture(output_name)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("input frames = " , length )





















