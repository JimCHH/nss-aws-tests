from extraction import *
from visualization import *

import sys

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
    center_list = []
    
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

        index = np.argwhere(temp == 1)
        if index.shape != (0, 2):
            x_center = np.average(index[:,0])
            y_center = np.average(index[:,1])
            center_list.append([x_center, y_center])
        else:
            center_list.append([0, 0])

    t4 = time.time()
    t.append(t4-t3)
    
    return inputImg_BK, t, center_list

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

save_time = []
fps = []

import logging, os
logging.basicConfig(
    level=logging.INFO, 
    filename=os.path.join('/home/ubuntu', 'nss-aws-tests', 'auto.py.log'),
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s %(levelname).3s] %(message)s')

date_site_patient = sys.argv[1]
source = f'/home/ubuntu/S3/{date_site_patient.split("_")[1]}/Result/{date_site_patient}'
for mp4_path in glob.glob(f'{source}/*.mp4'):
    if mp4_path[-9:-4] == 'Test4':
        continue

    cap = cv2.VideoCapture(mp4_path)
    print(f'\nTesting U-Net on {cap.get(cv2.CAP_PROP_FRAME_COUNT):.0f} frames of {mp4_path.split("/")[-1]}')

    output_mp4_path = mp4_path[:-4] + '_OUTPUT.MP4' # testing_video.mp4
    # out = cv2.VideoWriter(output_mp4_path, fourcc, 210, (384, 144), 0)
    frame_count = 0

    batch_size = 16
    counter = 0
    images = []
    temp_image = []
    center_list = []
    left_center_list_H = []
    left_center_list_V = []
    right_center_list_H = []
    right_center_list_V = []
    t = []
    data = {'Left':{'Horizontal':{}, 'Vertical':{}}, 'Right':{'Horizontal':{}, 'Vertical':{}}, 'Timestamps':{}}

    while cap.isOpened():
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
            images, times, center_list = Unet_test(images, model, Size_X, Size_Y)
            save_time.append(times)
            for i in range(batch_size//2):
                left_result = images[i*2].astype(np.uint8)    
                right_result = images[i*2+1].astype(np.uint8)
                result = np.concatenate((left_result, right_result), axis = 1)
                # out.write(result)
                # cv2.imshow('frame', result)
                left_center_list = (center_list[i*2][0], center_list[i*2][1])
                right_center_list = (center_list[i*2+1][0], center_list[i*2+1][1])
    
                left_center_list_H.append(left_center_list[1])
                left_center_list_V.append(left_center_list[0])
                right_center_list_H.append(right_center_list[1])
                right_center_list_V.append(right_center_list[0])
                t.append(frame_count/2+i)

            frame_count +=batch_size
            images = []
            counter = 0

            data['Right'].update({'Horizontal': np.array(left_center_list_H)}) # this Right is for Right eye (at the left side)
            data['Right'].update({'Vertical': np.array(left_center_list_V)})  
            data['Left'].update({'Horizontal': np.array(right_center_list_H)}) # this Left is for Left eye (at the right side)
            data['Left'].update({'Vertical': np.array(right_center_list_V)})
            data.update({'Timestamps': np.array(t)})

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break       
        
    end_time = time.time()
    print(f'Inference time: {end_time - start_time:.0f} s')
    print(f'Inference speed: {frame_count / (end_time - start_time):.0f} fps')

    cap.release()
    # out.release()
    # print(f'{cv2.VideoCapture(output_mp4_path).get(cv2.CAP_PROP_FRAME_COUNT)} frames in OUTPUT.MP4')

    cv2.destroyAllWindows()

    # with open(mp4_path[:-4] + '_unet_pixel_API.pkl', 'wb') as f:
    #     pickle.dump(data, f)

    try:
        extraction(mp4_path, data)
        visualization(mp4_path[:-4] + '_sp_dataset_API.pkl')
    except Exception as e:
        logging.info(e)
    else:
        logging.info(mp4_path.split('/')[-1][:-4])