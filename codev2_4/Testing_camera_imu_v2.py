# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:58:08 2021

@author: lab70929
"""
import torch
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet
import parameters
import cv2
import glob
import pandas as pd
device = torch.device("cuda")
'''
count = 0
for i in values:
    if len(i) != 16:
        print(count)
    count +=1    
''' 
def read_csv(path):
    data = pd.read_csv(path, delimiter="\t")
    length = data.shape[0]
    user_info = data.values[0][0].split(",")
    while('' in user_info):
        user_info.remove('')
        
    title = data.values[3][0].split(",")
    values = []
    crit_point = []
    for i in range(4,length):
        temp = data.values[i][0].split(",")
        while('' in temp):
            temp.remove('')
        
        if float(temp[13]) != 0:
            if float(temp[13]) == 1:
                prev_index = data.values[i-1][0].split(",")[14]
                if temp[14] == prev_index:
                    is_right = True
                else:
                    is_right = False
                crit_point.append([i-4  ,temp[-3:], is_right])
            temp = temp[:-3]
        values.append(temp)     
    
    values = np.array(values)
    
    
    return user_info, title, values, crit_point


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
        
        index = np.argwhere(temp == 1)
        if index.shape != (0, 2):
            x_center = np.average(index[:,0])
            y_center = np.average(index[:,1])
            center_list.append([x_center, y_center])
        else:
            center_list.append([0, 0])
            
        inputImg_BK[i] = gt_temp
        
        
       
    
    return inputImg_BK,center_list

def smooth_data(np_data,rate):
    if rate % 2 ==1:
        rate+=1
    length = np_data.shape[0]
    new_data = np.zeros(length)
    top = np.ones(rate//2) * np_data[0]
    bottom = np.ones(rate//2) * np_data[-1]
    np_data = np.concatenate((top,np_data,bottom))
    for i in range (length):
        new_data[i] = np.average(np_data[i:i+rate])
    return new_data

def get_time(TimeStamp):
    
    result = []

    for data in TimeStamp:
        data = data[-12:]
        hour = float(data[:2])
        minute = float(data[3:5])
        second = float(data[6:])       
        result.append(hour*60*60 + minute*60 + second)
    result = np.array(result)
    result -= result[0]
    return result

def convert_time_string(time_string):
    
    time_string = time_string[-12:]
    hour = float(time_string[:2])
    minute = float(time_string[3:5])
    second = float(time_string[6:])       
    time_value = hour*60*60 + minute*60 + second
    
    return time_value


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


#roi
left_roi = [[90,120],[474,360]]
right_roi = [[738,130],[1122,330]]


fourcc = cv2.VideoWriter_fourcc(*'mp4v')


save_time = []
fps = []

#read data
#video_folder = "C:/Users/lab70929/Downloads/usbcam_s3/usbcam_s/usbcam/bin/Debug/success/"

video_folder = os.getcwd()[:-6] + "Result"
output_folder = "video/"
video_list = glob.glob(video_folder+ "/*/" + "*.mp4")

#imu_list = glob.glob(video_folder+"*.csv")

for video_name in video_list:
    
    folder = output_folder + video_name.split("\\")[-2]
    try:
        os.mkdir(folder)
    except:
        print("資料夾已存在")
    
    imu_name = video_name.replace(".mp4", ".csv")
    
    output_video_name = folder + "/" + video_name.split("\\")[-1]
    output_csv_name = folder + "/" + imu_name.split("\\")[-1]
    output_plot_name = output_csv_name.replace(".csv", ".png")
    
    out = cv2.VideoWriter(output_video_name, fourcc, 210, (384, 144), 0)
    cap = cv2.VideoCapture(video_name)
    
    user_info, title, values, crit_point = read_csv(imu_name)
    
    HeadPos = values[:,1].astype(np.float32)
    HeadVel = values[:,2].astype(np.float32)
    TimeStamp = values[:,0]
    imu_x = get_time(TimeStamp)
    
    position = []
    position.append([[60,60],[60,60]])
    
    frame_counter = 0
    start_time = time.time()
    
    while(cap.isOpened()):
        
        images =[]        
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
        
        images, index= Unet_test(images, model, Size_X, Size_Y)
        if index[0][0] == 0:
            index[0] = position[frame_counter][0]
        
        if index[1][0] == 0:
            index[1] = position[frame_counter][1]        
        
        position.append(index)
        
        left_result = images[0].astype(np.uint8)
        cv2.circle(left_result,(round(index[0][1]), round(index[0][0])), 10, (0, 255, 255), 3)
        text = "x = " + str(index[0][1]) + " " + "y = " + str(index[0][0])
        cv2.putText(left_result, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        right_result = images[1].astype(np.uint8)
        cv2.circle(right_result,(round(index[1][1]), round(index[1][0])), 10, (0, 255, 255), 3)
        text = "x = " + str(index[1][1]) + " " + "y = " + str(index[1][0])
        cv2.putText(right_result, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)            
        result = np.concatenate((left_result, right_result), axis = 1)
        
        
        out.write(result)           
        cv2.imshow('frame', result)
#            cv2.imwrite(str(frame_counter) + '.jpg', result)
        frame_counter +=1        
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            break       

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    
    fps = frame_counter / (time.time() - start_time) 
    print("fps = ", fps)
    
    position = np.array(position)
    left = np.array(position[:,0,:])
    right = np.array(position[:,1,:])

    left_y_initial, left_x_initial = np.average(left, axis = 0)
    right_y_initial, right_x_initial = np.average(right, axis = 0)

    left[:, 0] = (left[:, 0] - left_y_initial) *0.66
    left[:, 1] = (left[:, 1] - left_x_initial) *0.66
    right[:, 0] = (right[:, 0] - right_y_initial) *0.66
    right[:, 1] = (right[:, 1] - right_x_initial) *0.66

    left[0, :] = left[1, :]
    right[0, :] = right[1, :]

    
    left_gradient = np.gradient(left, axis=0)*210
#    left_gradient = np.linalg.norm(left_gradient, axis = 1)
    left_gradient = left_gradient[:,1]

    left_gradient = smooth_data(left_gradient, 5)
    
    right_gradient = np.gradient(right, axis=0)*210
#    right_gradient = np.linalg.norm(right_gradient, axis = 1)
    right_gradient = right_gradient[:,1]
    right_gradient = smooth_data(right_gradient, 5)    
    
    x = np.arange(left_gradient.shape[0]) / 210
    
    plt.figure(frame_counter)
    plt.plot(x, left_gradient, label = "left_eye")
    plt.plot(x, right_gradient, label = "right_eye")
    plt.plot(imu_x, np.array(HeadVel, dtype=np.float), label = "imu_data")
    
    time_index = []
    for ind in crit_point:
        time_index.append(imu_x[ind[0]])
    for xc in time_index:
        plt.axvline(x=xc, c = 'b')    
        
    plt.legend()
    plt.title(output_plot_name)
    plt.xlabel("time(s)")
    plt.ylabel("degree")    
    plt.savefig(output_plot_name)
    count = 0
    left_count = 0
    right_count = 0
    sec = 2
    x = np.arange(210*sec)/210
    for t in crit_point:
        if t[2] == True:
            plt.figure(1)
            if right_count == 1:
                plt.legend()    
            right_count +=1
        if t[2] == False:
            plt.figure(2)
            if left_count == 1:
                plt.legend()                
            left_count +=1
            
        ind = int(time_index[count] * 210)
        plt.plot(x, left_gradient[ind - 210*sec :ind], c = 'b', label = "left_eye")
        plt.plot(x, right_gradient[ind - 210*sec :ind], c = 'r',  label = "right_eye")
        
        imu_left_index = np.argmin(np.abs(imu_x - (time_index[count] - sec)))
        imu_right_index = np.argmin(np.abs(imu_x - time_index[count]))
        
        plt.plot(imu_x[imu_left_index : imu_right_index] - imu_x[imu_left_index], HeadVel[imu_left_index : imu_right_index], c = 'black', label = "imu_data")
        count+=1
        
        
        

    plt.figure(1)
    plt.title("inpulse_right, N=" + str(right_count))
    plt.xlabel("t (s)")
    plt.ylabel("Velociety (degree / s)")    
    
    plt.figure(2)
    plt.title("inpulse_left, N=" + str(left_count))
    plt.xlabel("t (s)")
    plt.ylabel("Velociety (degree / s)")    

#    output = np.stack((left_gradient, right_gradient))
#    output = np.transpose(output)
#    np.savetxt(output_csv_name, output, delimiter=',')
    










