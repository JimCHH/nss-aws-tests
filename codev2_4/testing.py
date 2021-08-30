# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 01:37:49 2021

@author: lab70929
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch
import parameters
from unet import UNet
import os
device = torch.device("cuda")






def Unet_test(images, model, Size_X, Size_Y):
    batch_size = len(images)
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
        

    
    tf_images = torch.from_numpy(inputImg)
    tf_images = tf_images.to(device)
    output = model(tf_images)
    output_bk = output[:, 0].clone().detach().cpu().numpy()
    
    
    
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
        gt_temp[temp != 1] = 0
        index = np.argwhere(temp == 1)
        if index.shape != (0, 2):
            x_center = np.average(index[:,0])
            y_center = np.average(index[:,1])
            center_list.append([x_center, y_center])
        else:
            center_list.append([0, 0])
            
        inputImg_BK[i] = gt_temp
        
        
       
    
    return inputImg_BK,center_list





test_eye_dataset_list = glob.glob("test_eye_dataset/*")
ground_truth_list = glob.glob("ground_truth/*")
test_data = []
ground_truth = []
name_list = []
for i in range(len(test_eye_dataset_list)):
    
    data_list = glob.glob(test_eye_dataset_list[i]+"/*")
    gh_list = glob.glob(ground_truth_list[i]+"/*")
    name_list += gh_list
    for j in range(len(gh_list)):
        test_image = cv2.imread(data_list[j], 0)
        left_test_image = test_image[:,:640]
        right_test_image = test_image[:,640:]
       
        gh_image = cv2.imread(gh_list[j], 0)
 
        left_gh_image = gh_image[:,:640]
        right_gh_image = gh_image[:,640:]        
        left_gh_image = cv2.resize(left_gh_image, (192, 144), interpolation=cv2.INTER_CUBIC)
        right_gh_image = cv2.resize(right_gh_image, (192, 144), interpolation=cv2.INTER_CUBIC)
        left_gh_image[left_gh_image<200] = 0
        left_gh_image[left_gh_image>200] = 255               
        right_gh_image[right_gh_image<200] = 0
        right_gh_image[right_gh_image>200] = 255    
        
        test_data.append(left_test_image)
        test_data.append(right_test_image)
        
        ground_truth.append(left_gh_image)
        ground_truth.append(right_gh_image)
        
#load model
name_load_model = 'trained_model/UNet/'
cross_val_num = 18
Size_X = parameters.Size_X
Size_Y = parameters.Size_Y
model = UNet(n_channels=1, n_classes=1, bilinear=True)
if os.path.exists(name_load_model):
    load_saved_model_name = parameters.find_latest_model_name(name_load_model, cross_val_num)
    model.load_state_dict(torch.load(load_saved_model_name))
    print(parameters.C_GREEN + 'Check point Successfully Loaded' + parameters.C_END)
else:
    print(parameters.C_RED + 'Check point Not Found' + parameters.C_END)

model.eval()
model.to(device)

result = []
result_center = []
batch_size = 10
for i in range(len(test_data)//batch_size):
    inputImg_BK,center_list = Unet_test(test_data[i*10:i*10+10], model, Size_X, Size_Y)
    result += inputImg_BK
    result_center += center_list

ground_truth_center = []
error_list = []
IOU_list = []
I = 0
U = 0

for i in range(len(result)):
    A = result[i] > 0
    B = ground_truth[i] > 0
    C = np.logical_and(A,B)
    D = np.logical_or(A,B)
    if np.sum(D) == 0:
        ground_truth_center.append([0, 0])    
        print(i)
        continue
    I += np.sum(C)
    U += np.sum(D)
    IOU =  np.sum(C) / np.sum(D)
    
    back_ground = cv2.resize(test_data[i],(192, 144), interpolation=cv2.INTER_CUBIC)
    back_ground = cv2.cvtColor(back_ground, cv2.COLOR_GRAY2RGB)
    
    copy_back_ground = back_ground.copy()
    
    
    
    
    
    index = np.argwhere(B == True)
    if index.shape != (0, 2):
        x_center = np.average(index[:,0])
        y_center = np.average(index[:,1])
        ground_truth_center.append([x_center, y_center])
    else:
        ground_truth_center.append([0, 0])    
    
    
    error = np.linalg.norm((x_center - result_center[i][0], y_center - result_center[i][1]))
    
    back_ground[A,0] = 0
    back_ground[A,1] = 0
    back_ground[A,2] = 255

    back_ground[B,0] = 0
    back_ground[B,1] = 255
    back_ground[B,2] = 0

    back_ground[C,0] = 255
    back_ground[C,1] = 255
    back_ground[C,2] = 255
    
    predict_image = copy_back_ground.copy()
    predict_image[A,0] = 0
    predict_image[A,1] = 0
    predict_image[A,2] = 255    
    
    ground_truth_image = copy_back_ground.copy()
    ground_truth_image[B,0] = 0
    ground_truth_image[B,1] = 255
    ground_truth_image[B,2] = 0        
    
    


    result_image_temp = np.concatenate((back_ground, predict_image), axis=1)
    result_image = np.concatenate((ground_truth_image, copy_back_ground), axis=1)
    result_image = np.concatenate((result_image_temp, result_image), axis=0)
    
    cv2.putText(result_image, "score = " + str(IOU)[:6], (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,(255, 0, 0))
    cv2.putText(result_image, "error = " + str(error)[:6], (10, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7,(255, 0, 0))
    
    plt.imshow(result_image)
    save_name = name_list[i//2].replace("ground_truth","testing_result")
    save_name = save_name[:-4] + "_"+ str(i%2) + save_name[-4:]
    plt.imsave(save_name, result_image)
    error_list.append(error)
    IOU_list.append(IOU)    


results = np.array([error_list, IOU_list])
results = np.transpose(results)
np.savetxt("testing_result.csv", results, delimiter=',', header = "IOU,error")












