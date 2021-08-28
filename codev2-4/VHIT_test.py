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
import xmltodict
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
device = torch.device("cuda")
data_length = 400
def read_csv(path):
    
    #parameter setting
    back_range = [int(70*-3), int(70*-0.5)]
    foreward_time = 0.15
    backward_time = 0.25
    
    
    
    
    splitted_data = []
    data = pd.read_csv(path, delimiter="\t")
    length = data.shape[0]

    user_info = data.values[0][0].split(",")
    while('' in user_info):
        user_info.remove('')
        
    title = data.values[3][0].split(",")
    
    time_list = []
    data_array = []
    for i in range(4,length):
        time_list.append(data.values[i][0].split(",")[0])
        data_array.append(data.values[i][0].split(",")[2])
        
    time_array = get_time(time_list)
    data_array = np.array(data_array).astype(np.float32)
    
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
                    
                data_temp = []
                if is_right:
                    peak_time = time_array[np.argmin(data_array[i-4 + back_range[0] : i-4 + back_range[1]]) + i-4 + back_range[0]]
                else:
                    peak_time = time_array[np.argmax(data_array[i-4 + back_range[0] : i-4 + back_range[1]]) + i-4 + back_range[0]]
                    
                start_time = peak_time - foreward_time
                end_time = peak_time + backward_time
                
                start_index = np.argmin(np.abs(time_array - start_time))
                end_index = np.argmin(np.abs(time_array - end_time))
                
                start_time_string = time_list[start_index]
                
                for ind in range(start_index,end_index):
                    temp = data.values[ind +4][0].split(",")
                    while('' in temp):
                        temp.remove('')
                    data_temp.append(np.array(temp[:13]))
                splitted_data.append([time_array[start_index:end_index]  ,data_temp, is_right, start_time_string]) # [time, data, is_right]
    
    
    return user_info, title, splitted_data

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

def create_dict_templete():
    Left_EyeVelocitySamples = ""
    Right_EyeVelocitySamples = ""
    Gain = ""
    HeadVelocitySamples = ""
    IsDirectionLeft = ""
    Timestamp = ""
    HIImpulse = [{"Left_EyeVelocitySamples" : Left_EyeVelocitySamples, "Right_EyeVelocitySamples": Right_EyeVelocitySamples, "Timestamp" : Timestamp, "Gain" : Gain, "HeadVelocitySamples" : HeadVelocitySamples, "IsDirectionLeft" : IsDirectionLeft}]
    NumAcceptedLeftImpulses = ""
    NumAcceptedRightImpulses = ""
    PatientUID = ""
    HITest = {"HIImpulse" : HIImpulse, "NumAcceptedLeftImpulses" : NumAcceptedLeftImpulses, "NumAcceptedRightImpulses": NumAcceptedRightImpulses, "PatientUID" : PatientUID}
    FrameRate = ""
    FileNameWithPath = ""
    VideoUID = ""
    ICSVideo = {"FileNameWithPath" : FileNameWithPath, "FrameRate" : FrameRate, "VideoUID" : VideoUID}
    PatientID = ""
    Date = ""
    Doctor = ""
    Device = ""
    Name = ""
    Exam = ""
    Mode = ""
    ICSPatient = {"HITest" : HITest, "ICSVideo" : ICSVideo, "PatientID" : PatientID, "Date" : Date, "Doctor" : Doctor, "Device" : Device, "Name" : Name, "Exam" : Exam, "Mode" : Mode}
    ICSSuiteDBPMRDataSet = {"ICSPatient" : ICSPatient}
    templete = {"ICSSuiteDBPMRDataSet" : ICSSuiteDBPMRDataSet}
    return templete

def convert2str(data):
    
    temp = np.array2string(data, separator = ";", formatter={'float_kind':lambda x: "%.5f" % x})
    string = temp.replace(".", ",")
    string = temp.replace(" ","0")
    string = string[1:]
    string = string[:-1]
    return string

def list2array(data_list):
    np_array = np.array([])
    
    for i in range(len(data_list)):
        np_array = np.concatenate((np_array, data_list[i]), axis=0)
    
    return np_array

cm = plt.cm.get_cmap('rainbow')
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

# video_folder = os.getcwd()[:-6] + "Result"
source = f'{os.path.expanduser("~")}/s3/{date_site_patient.split("_")[1]}/{date_site_patient}'
target = f'{os.path.expanduser("~")}/s3/Reports'
# video_list = glob.glob(video_folder+ "/*/" + "*.mp4")
gain_type = 0

#imu_list = glob.glob(video_folder+"*.csv")
#mp4_path = video_list[11]
for mp4_path in glob.glob(f'{source}/*.mp4'):
    start_time = time.time()
    print(f'{mp4_path} 進行瞳孔追蹤。。。')
    date_site_patient_path = f'{target}/{mp4_path.split("/")[-2]}'
    try:
        os.mkdir(date_site_patient_path)
        print(f'新增 {date_site_patient_path}')
    except:
        print(f'{date_site_patient_path} 已存在')
    
    vhit_name = mp4_path.replace(".mp4", ".csv")
    
    user_info, title, splitted_data = read_csv(vhit_name)
    if len(splitted_data) == 0 :
        continue  
    output_mp4 = f'{date_site_patient_path}/{mp4_path.split("/")[-1]}'
    output_xml = output_mp4[:-4] + ".xml"#output_mp4.replace(".mp4",".xml")
    output_right = output_mp4[:-4] + "_right.pdf" 
    output_left  = output_mp4[:-4] + "_left.pdf" 
    
    out = cv2.VideoWriter(output_mp4, fourcc, 210, (384, 144), 0)
    cap = cv2.VideoCapture(mp4_path)
    
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    
    
    position_data = []

    for number in range(len(splitted_data)):
        
    
        HeadPos = np.array(splitted_data[number][1])[:,1].astype(np.float32)
        HeadVel = np.array(splitted_data[number][1])[:,2].astype(np.float32)
        imu_x = np.array(splitted_data[number][0]).astype(np.float32)
    
        position = []
        position.append([[60,60],[60,60]])
        
        frame_counter = 0
        
        video_start_index = int(imu_x[0] * 210)
        video_end_index = int(imu_x[-1] * 210)
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_index)
        frame_number = video_end_index - video_start_index
        
        for i in range(frame_number):
            
            images = []        
            ret, frame = cap.read()
            if ret == False:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if left_roi == []:
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
            cv2.circle(left_result, (int(round(index[0][1])), int(round(index[0][0]))), 10, (0, 255, 255), 3)
            text = "x = " + str(index[0][1]) + " " + "y = " + str(index[0][0])
            cv2.putText(left_result, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            right_result = images[1].astype(np.uint8)
            cv2.circle(right_result, (int(round(index[1][1])), int(round(index[1][0]))), 10, (0, 255, 255), 3)
            text = "x = " + str(index[1][1]) + " " + "y = " + str(index[1][0])
            cv2.putText(right_result, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)            
            result = np.concatenate((left_result, right_result), axis = 1)
            
            
            out.write(result)           
            # cv2.imshow('frame', result)
    #            cv2.imwrite(str(frame_counter) + '.jpg', result)
            frame_counter +=1        
            
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                
                break       
        position_data.append(position)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    
    total_time = time.time() - start_time
    print("total_time = ", total_time)    
    #-------------------------------------------------------------------------
    frame_index_0 = 0
    frame_index_40 = int(40*210/1000)
    frame_index_60 = int(60*210/1000)
    frame_index_80 = int(80*210/1000)
    frame_index_100 = int(100*210/1000)
    
    imu_index_0 = 0
    imu_index_40 = int(40*70/1000)
    imu_index_60 = int(60*70/1000)
    imu_index_80 = int(80*70/1000)
    imu_index_100 = int(100*70/1000)
    
    
    left_fig, left_axs = plt.subplots(2,3,figsize=(16,9),dpi = 100, gridspec_kw={'height_ratios': [2, 1]})
    right_fig, right_axs = plt.subplots(2,3,figsize=(16,9),dpi = 100, gridspec_kw={'height_ratios': [2, 1]})

    left_axs_bbox = left_axs[0,2].get_position()
    p_x = left_axs_bbox.p0[0]
    p_x1 = (left_axs_bbox.p0[0] + left_axs_bbox.p1[0]) /2
    p_y = left_axs_bbox.p0[1]
    w = left_axs_bbox.width / 2
    h = left_axs_bbox.height /2
    
    ax_left_combine_2 = left_fig.add_axes([p_x, p_y, w*2, h+0.001])   
    ax_left_combine = left_fig.add_axes([p_x, p_y, w*2, h])   
    ax_left_right = left_fig.add_axes([p_x, p_y, w, h])
    ax_left_left = left_fig.add_axes([p_x1, p_y, w, h])

    right_axs_bbox = right_axs[0,2].get_position()
    p_x = right_axs_bbox.p0[0]
    p_x1 = (right_axs_bbox.p0[0] + right_axs_bbox.p1[0]) /2
    p_y = right_axs_bbox.p0[1]
    w = right_axs_bbox.width / 2
    h = right_axs_bbox.height /2
    
    ax_right_combine_2 = right_fig.add_axes([p_x, p_y, w*2, h+0.001])   
    ax_right_combine = right_fig.add_axes([p_x, p_y, w*2, h])   
    ax_right_right = right_fig.add_axes([p_x, p_y, w, h])
    ax_right_left = right_fig.add_axes([p_x1, p_y, w, h])

        
    left_count = 0
    right_count = 0    
    
    total_left_vhit = 0
    total_right_vhit = 0
    
    left_gain = []
    right_gain = []
    
    left_HeadVel_list = []
    right_HeadVel_list = []
    
    left_gradient_list = []
    right_gradient_list = []
    Output_Gain_list = []
    
    left_eye_left_list = []
    left_eye_right_list = []
    right_eye_left_list = []
    right_eye_right_list = []
    
    left_Vol_list = []
    right_Vol_list = []

    for i in range(len(splitted_data)):
        if splitted_data[i][2] == True:
            total_right_vhit += 1
        else:
            total_left_vhit += 1
    
    for i in range(len(splitted_data)):
        HeadVel = np.array(splitted_data[i][1])[:,2].astype(np.float32)
        imu_x = np.array(splitted_data[i][0]).astype(np.float32)
        imu_x = imu_x*1000 #time scale > s to  ms
        new_imu_x = imu_x - imu_x[0]
    
            
        position = np.array(position_data[i])
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
        
        left_gradient_list.append(left_gradient)
        right_gradient_list.append(right_gradient)
        
    
        x = np.arange(left_gradient.shape[0]) / 210
        x = x*1000 #time scale > s to  ms
        if splitted_data[i][2] == True:
            if right_count == 1:
                left_axs[0,0].legend()   
                right_axs[0,0].legend()   
                
            left_axs[0,0].plot(x, left_gradient, label = "left_eye", c = cm(right_count/total_right_vhit))
            left_axs[0,0].plot(new_imu_x, HeadVel, label = "imu_data", c = 'black')
            
            right_axs[0,0].plot(x, right_gradient, label = "right_eye",  c = cm(right_count/total_right_vhit))
            right_axs[0,0].plot(new_imu_x, HeadVel, label = "imu_data", c = 'black')
            
            
            
            
            gain_index = np.argmin(np.abs(new_imu_x - 100))
            gain_x = new_imu_x[:gain_index]
            gain_Vel = HeadVel[:gain_index]
            f = interp1d(x, left_gradient)
            left_eye_gain = np.abs((f(gain_x)+1) / (gain_Vel+1))
            left_axs[1,0].plot(gain_x, left_eye_gain)   
            
            right_eye_right_list.append(f(gain_x))
            
            f = interp1d(x, right_gradient)
            right_eye_gain = np.abs((f(gain_x)+1) / (gain_Vel+1))
            right_axs[1,0].plot(gain_x, right_eye_gain)
            left_eye_right_list.append(f(gain_x))
            right_Vol_list.append(gain_Vel)
    

            gain_40 = [left_gradient[frame_index_40] / HeadVel[imu_index_40],  right_gradient[frame_index_40] / HeadVel[imu_index_40]]
            gain_60 = [left_gradient[frame_index_60] / HeadVel[imu_index_40],  right_gradient[frame_index_60] / HeadVel[frame_index_60]]
            gain_80 = [left_gradient[frame_index_80] / HeadVel[imu_index_80],  right_gradient[frame_index_80] / HeadVel[imu_index_80]]
            gain_0_100 = [np.average(left_gradient[frame_index_0:frame_index_100]) / np.average(HeadVel[imu_index_0:imu_index_100]), \
                          np.average(right_gradient[frame_index_0:frame_index_100]) / np.average(HeadVel[imu_index_0:imu_index_100])]
            
            temp_list = [gain_40, gain_60, gain_80, gain_0_100]
            right_gain.append(temp_list)                

            right_HeadVel_list.append([HeadVel[imu_index_40], HeadVel[imu_index_60], HeadVel[imu_index_80], np.average(HeadVel[imu_index_0:imu_index_100])])
            Output_Gain_list.append(temp_list[gain_type])
            right_count +=1
            
    
        if splitted_data[i][2] == False:
            
            if left_count == 1:
                left_axs[0,1].legend()   
                right_axs[0,1].legend()   
                 
            left_axs[0,1].plot(x, left_gradient, label = "left_eye", c = cm(left_count/total_left_vhit))        
            left_axs[0,1].plot(new_imu_x, HeadVel, label = "imu_data", c = 'black')
            
            right_axs[0,1].plot(x, right_gradient, label = "right_eye",  c = cm(left_count/total_left_vhit))
            right_axs[0,1].plot(new_imu_x, HeadVel, label = "imu_data", c = 'black')
            
            gain_index = np.argmin(np.abs(new_imu_x - 100))
            gain_x = new_imu_x[:gain_index]
            gain_Vel = HeadVel[:gain_index]
            f = interp1d(x, left_gradient)
            left_eye_gain = np.abs((f(gain_x)+1) / (gain_Vel+1))
            left_axs[1,1].plot(gain_x, left_eye_gain)   
            left_eye_left_list.append(f(gain_x))
            f = interp1d(x, right_gradient)
            right_eye_gain = np.abs((f(gain_x)+1) / (gain_Vel+1))
            right_axs[1,1].plot(gain_x, right_eye_gain)
            right_eye_left_list.append(f(gain_x))
            left_Vol_list.append(gain_Vel)

            gain_40 = [left_gradient[frame_index_40] / HeadVel[imu_index_40],  right_gradient[frame_index_40] / HeadVel[imu_index_40]]
            gain_60 = [left_gradient[frame_index_60] / HeadVel[imu_index_40],  right_gradient[frame_index_60] / HeadVel[frame_index_60]]
            gain_80 = [left_gradient[frame_index_80] / HeadVel[imu_index_80],  right_gradient[frame_index_80] / HeadVel[imu_index_80]]
        
            gain_0_100 = [np.average(left_gradient[frame_index_0:frame_index_100]) / np.average(HeadVel[imu_index_0:imu_index_100]), \
                          np.average(right_gradient[frame_index_0:frame_index_100]) / np.average(HeadVel[imu_index_0:imu_index_100])]
            
            
            temp_list = [gain_40, gain_60, gain_80, gain_0_100]
            left_gain.append(temp_list)                
            left_HeadVel_list.append([HeadVel[imu_index_40], HeadVel[imu_index_60], HeadVel[imu_index_80], np.average(HeadVel[imu_index_0:imu_index_100])])
            
            Output_Gain_list.append(temp_list[gain_type])
            left_count +=1      

    left_axs[0,0].set_title("Inpulse_right, Left_eye, N=" + str(right_count))
    left_axs[0,0].set_xlabel("Time [ms]")
    left_axs[0,0].set_ylabel("Velociety [degree / s]")  
    left_axs[0,0].set_xlim(0, data_length)
    left_axs[0,0].set_ylim(-300,300)
    left_axs[0,0].grid(ls='--')
    
    right_axs[0,0].set_title("Inpulse_right, Right_eye, N=" + str(right_count))
    right_axs[0,0].set_xlabel("Time [ms]")
    right_axs[0,0].set_ylabel("Velociety [degree / s]")  
    right_axs[0,0].set_xlim(0, data_length)
    right_axs[0,0].set_ylim(-300,300)
    right_axs[0,0].grid(ls='--')
    
    left_axs[0,1].set_title("Inpulse_left, Left_eye, N=" + str(left_count))
    left_axs[0,1].set_xlabel("Time [ms]")
    left_axs[0,1].set_ylabel("Velociety [degree / s]")  
    left_axs[0,1].set_xlim(0 ,data_length)
    left_axs[0,1].set_ylim(-300,300)
    left_axs[0,1].grid(ls='--')
    
    right_axs[0,1].set_title("Inpulse_left, Right_eye, N=" + str(left_count))
    right_axs[0,1].set_xlabel("Time [ms]")
    right_axs[0,1].set_ylabel("Velociety [degree / s]")  
    right_axs[0,1].set_xlim(0 ,data_length)
    right_axs[0,1].set_ylim(-300,300)
    right_axs[0,1].grid(ls='--')
    
    left_gain = np.array(left_gain)
    left_gain = np.abs(left_gain)
    
    right_gain = np.array(right_gain)
    right_gain = np.abs(right_gain)
    total_gain = [right_gain, left_gain]
    for i in range(2):
        left_axs[1,i].set_xlabel("Time [ms]")
        left_axs[1,i].set_ylabel("Gain")  
        left_axs[1,i].set_ylim(0,1.5)
        left_axs[1,i].set_xlim(0,data_length)
        left_axs[1,i].grid(ls='--')
        
        left_axs[1,i].text(125 ,1.49 , "Gain", fontsize = 14, va = 'top')
        left_axs[1,i].text(125 ,1.3 , "40 ms :", fontsize = 14, va = 'top')
        left_axs[1,i].text(125 ,1.1 , "60 ms :", fontsize = 14, va = 'top')
        left_axs[1,i].text(125 ,0.9 , "80 ms :", fontsize = 14, va = 'top')
        left_axs[1,i].text(125 ,0.7 , "0~100 ms :", fontsize = 14, va = 'top')
        
        left_axs[1,i].text(250 ,1.3 , str(np.mean(total_gain[i][:, 0, 0]))[:6], fontsize = 14, va = 'top')
        left_axs[1,i].text(250 ,1.1 , str(np.mean(total_gain[i][:, 1, 0]))[:6], fontsize = 14, va = 'top')
        left_axs[1,i].text(250 ,0.9 , str(np.mean(total_gain[i][:, 2, 0]))[:6], fontsize = 14, va = 'top')
        left_axs[1,i].text(250 ,0.7 , str(np.mean(total_gain[i][:, 3, 0]))[:6], fontsize = 14, va = 'top')
        
        right_axs[1,i].set_xlabel("Time [ms]")
        right_axs[1,i].set_ylabel("Gain")  
        right_axs[1,i].set_ylim(0,1.5)
        right_axs[1,i].set_xlim(0,data_length)
        right_axs[1,i].grid(ls='--')

        right_axs[1,i].text(125 ,1.49 , "Gain", fontsize = 14, va = 'top')
        right_axs[1,i].text(125 ,1.3 , "40 ms :", fontsize = 14, va = 'top')
        right_axs[1,i].text(125 ,1.1 , "60 ms :", fontsize = 14, va = 'top')
        right_axs[1,i].text(125 ,0.9 , "80 ms :", fontsize = 14, va = 'top')
        right_axs[1,i].text(125 ,0.7 , "0~100 ms :", fontsize = 14, va = 'top')
        
        right_axs[1,i].text(250 ,1.3 , str(np.mean(total_gain[i][:, 0, 1]))[:6], fontsize = 14, va = 'top')
        right_axs[1,i].text(250 ,1.1 , str(np.mean(total_gain[i][:, 1, 1]))[:6], fontsize = 14, va = 'top')
        right_axs[1,i].text(250 ,0.9 , str(np.mean(total_gain[i][:, 2, 1]))[:6], fontsize = 14, va = 'top')
        right_axs[1,i].text(250 ,0.7 , str(np.mean(total_gain[i][:, 3, 1]))[:6], fontsize = 14, va = 'top')
        
        
        


    left_HeadVel_array = np.array(left_HeadVel_list)
    left_HeadVel_array = np.abs(left_HeadVel_array)

    right_HeadVel_array = np.array(right_HeadVel_list)
    right_HeadVel_array = np.abs(right_HeadVel_array)
    
    left_axs[1,2].scatter(left_HeadVel_array[:, gain_type], left_gain[:, gain_type, 0] , c="b", label = "Left")
    right_axs[1,2].scatter(left_HeadVel_array[:, gain_type], left_gain[:, gain_type, 1] , c="b", label = "Left")
    
    left_axs[1,2].scatter(right_HeadVel_array[:, gain_type], right_gain[:, gain_type, 0] , c="r", label = "Right")
    right_axs[1,2].scatter(right_HeadVel_array[:, gain_type], right_gain[:, gain_type, 1] , c="r", label = "Right")    
    
    left_axs[1,2].set_ylabel("Gain")
    left_axs[1,2].set_xlabel("Head Velociety [degree / s]")  
    left_axs[1,2].legend()
    
    right_axs[1,2].set_ylabel("Gain")
    right_axs[1,2].set_xlabel("Head Velociety [degree / s]")      

    
    right_axs[1,2].legend()
    
    
    right_axs[0,2].axes.get_xaxis().set_visible(False)
    right_axs[0,2].axes.get_yaxis().set_visible(False)
    
    left_axs[0,2].axes.get_xaxis().set_visible(False)
    left_axs[0,2].axes.get_yaxis().set_visible(False)    
 
    add_height = 0.04
    fontsize = 12
    right_axs[0,2].text(0.01 ,0.99 , "Name :", fontsize = fontsize, va = 'top')
    right_axs[0,2].text(0.01 ,0.99 - add_height, "Date :", fontsize = fontsize, va = 'top')
    right_axs[0,2].text(0.01 ,0.99 - add_height*2, "Doctor :", fontsize = fontsize, va = 'top')
    right_axs[0,2].text(0.01 ,0.99 - add_height*3, "Device :", fontsize = fontsize, va = 'top')
    right_axs[0,2].text(0.01 ,0.99 - add_height*4, "Exam :", fontsize = fontsize, va = 'top')
    right_axs[0,2].text(0.01 ,0.99 - add_height*5, "Mode :", fontsize = fontsize, va = 'top')
    right_axs[0,2].text(0.01 ,0.99 - add_height*6, "Eye :", fontsize = fontsize, va = 'top')

    right_axs[0,2].text(0.33 ,0.99 , user_info[3], fontsize = fontsize, va = 'top')
    right_axs[0,2].text(0.33 ,0.99 - add_height, user_info[0], fontsize = fontsize, va = 'top')
    right_axs[0,2].text(0.33 ,0.99 - add_height*2, user_info[1], fontsize = fontsize, va = 'top')
    right_axs[0,2].text(0.33 ,0.99 - add_height*3, user_info[2], fontsize = fontsize, va = 'top')
    right_axs[0,2].text(0.33 ,0.99 - add_height*4, user_info[4], fontsize = fontsize, va = 'top')
    right_axs[0,2].text(0.33 ,0.99 - add_height*5, user_info[5], fontsize = fontsize, va = 'top')
    right_axs[0,2].text(0.33 ,0.99 - add_height*6, "Right_eye", fontsize = fontsize, va = 'top')    

    left_axs[0,2].text(0.01 ,0.99 , "Name :", fontsize = fontsize, va = 'top')
    left_axs[0,2].text(0.01 ,0.99 - add_height, "Date :", fontsize = fontsize, va = 'top')
    left_axs[0,2].text(0.01 ,0.99 - add_height*2, "Doctor :", fontsize = fontsize, va = 'top')
    left_axs[0,2].text(0.01 ,0.99 - add_height*3, "Device :", fontsize = fontsize, va = 'top')
    left_axs[0,2].text(0.01 ,0.99 - add_height*4, "Exam :", fontsize = fontsize, va = 'top')
    left_axs[0,2].text(0.01 ,0.99 - add_height*5, "Mode :", fontsize = fontsize, va = 'top')
    left_axs[0,2].text(0.01 ,0.99 - add_height*6, "Eye :", fontsize = fontsize, va = 'top')

    left_axs[0,2].text(0.33 ,0.99 , user_info[3], fontsize = fontsize, va = 'top')
    left_axs[0,2].text(0.33 ,0.99 - add_height, user_info[0], fontsize = fontsize, va = 'top')
    left_axs[0,2].text(0.33 ,0.99 - add_height*2, user_info[1], fontsize = fontsize, va = 'top')
    left_axs[0,2].text(0.33 ,0.99 - add_height*3, user_info[2], fontsize = fontsize, va = 'top')
    left_axs[0,2].text(0.33 ,0.99 - add_height*4, user_info[4], fontsize = fontsize, va = 'top')
    left_axs[0,2].text(0.33 ,0.99 - add_height*5, user_info[5], fontsize = fontsize, va = 'top')
    left_axs[0,2].text(0.33 ,0.99 - add_height*6, "Left_eye", fontsize = fontsize, va = 'top')    

    left_axs[0,2].spines["left"].set_visible(False)
    left_axs[0,2].spines["right"].set_visible(False)
    left_axs[0,2].spines["top"].set_visible(False)

    right_axs[0,2].spines["left"].set_visible(False)
    right_axs[0,2].spines["right"].set_visible(False)
    right_axs[0,2].spines["top"].set_visible(False)

    
    ax_left_right.set_xlim(300, 0)
    ax_left_left.set_xlim(0, 300)

    ax_left_right.set_ylim(0,350)
    ax_left_right.set_yticks(range(0,301,100))
    ax_left_left.set_ylim(0, 350)
    ax_left_left.set_yticks(range(150,301,150))
    ax_left_left.set_yticklabels(["0.5","1"])
    ax_left_left.tick_params(right= True, labelright = True)
    ax_left_right.set_xticks([])
    ax_left_left.set_xticks([])
    
    ax_left_combine.xaxis.tick_top()
    ax_left_combine.set_xlim(0, 200)
    ax_left_combine.set_xticks(range(0,201,50))
    ax_left_combine.set_xticklabels(["100","50", "0", "50", "100"])
    ax_left_combine.set_yticks([])
    ax_left_combine.set_xlabel("Gain Asymmetry [%]", )
    ax_left_combine.xaxis.set_label_position('top')
    ax_left_combine.set_title("Velocity regression")
    ax_left_right.set_ylabel("Eye [|degree/s|]", )
    ax_left_left.set_ylabel("Regression Slope = Gain", )
    ax_left_left.yaxis.set_label_position('right')
    
    ax_left_combine_2.set_yticks([])
    ax_left_combine_2.set_xlim(0, 600)
    ax_left_combine_2.set_xticks(range(0,601,100))
    ax_left_combine_2.set_xlabel("Head [|degree/s|]" )
    ax_left_combine_2.set_xticklabels(["300","200", "100", "0", "100", "200", "300"])
    
    ax_right_right.set_xlim(300, 0)
    ax_right_left.set_xlim(0, 300)

    ax_right_right.set_ylim(0,350)
    ax_right_right.set_yticks(range(0,301,100))
    ax_right_left.set_ylim(0, 350)
    ax_right_left.set_yticks(range(150,301,150))
    ax_right_left.set_yticklabels(["0.5","1"])
    ax_right_left.tick_params(right= True, labelright = True)
    ax_right_right.set_xticks([])
    ax_right_left.set_xticks([])
    
    ax_right_combine.xaxis.tick_top()
    ax_right_combine.set_xlim(0, 200)
    ax_right_combine.set_xticks(range(0,201,50))
    ax_right_combine.set_xticklabels(["100","50", "0", "50", "100"])
    ax_right_combine.set_yticks([])
    ax_right_combine.set_xlabel("Gain Asymmetry [%]", )
    ax_right_combine.xaxis.set_label_position('top')
    ax_right_combine.set_title("Velocity regression")
    ax_right_right.set_ylabel("Eye [|degree/s|]", )
    ax_right_left.set_ylabel("Regression Slope = Gain", )
    ax_right_left.yaxis.set_label_position('right')
    
    ax_right_combine_2.set_yticks([])
    ax_right_combine_2.set_xlim(0, 600)
    ax_right_combine_2.set_xticks(range(0,601,100))
    ax_right_combine_2.set_xlabel("Head [|degree/s|]" )
    ax_right_combine_2.set_xticklabels(["300","200", "100", "0", "100", "200", "300"])    


    right_eye_left_array = list2array(right_eye_left_list)
    right_eye_right_array = list2array(right_eye_right_list)
    left_eye_left_array = list2array(left_eye_left_list)
    left_eye_right_array = list2array(left_eye_right_list)

    right_Vol_array = list2array(right_Vol_list)
    left_Vol_array = list2array(left_Vol_list)

    right_eye_left_array = np.abs(right_eye_left_array)
    right_eye_right_array = np.abs(right_eye_right_array)    
    left_eye_left_array = np.abs(left_eye_left_array)
    left_eye_right_array = np.abs(left_eye_right_array)
    right_Vol_array = np.abs(right_Vol_array)
    left_Vol_array = np.abs(left_Vol_array)

    ax_left_right.scatter(right_Vol_array ,left_eye_right_array, c = "gray")
    ax_left_left.scatter(left_Vol_array ,left_eye_left_array, c = "gray")
    ax_right_right.scatter(right_Vol_array ,right_eye_right_array, c = "gray")
    ax_right_left.scatter(left_Vol_array ,right_eye_left_array, c = "gray")

    pred_x = np.arange(1,300)
    reg = LinearRegression().fit(right_Vol_array.reshape(-1, 1), left_eye_right_array.reshape(-1, 1))
    pred = reg.predict(pred_x.reshape(-1, 1))
    ax_left_right.plot(pred_x, pred, color  = "r")
    coef = reg.coef_
    ax_left_right.text(250,300,"coef = " + convert2str(coef[0]), color  = "r")
    
    reg = LinearRegression().fit(left_Vol_array.reshape(-1, 1), left_eye_left_array.reshape(-1, 1))
    pred = reg.predict(pred_x.reshape(-1, 1))
    ax_left_left.plot(pred_x, pred, color  = "b")    
    coef = reg.coef_
    ax_left_left.text(50,300,"coef = " + convert2str(coef[0]), color  = "b")
    
    reg = LinearRegression().fit(right_Vol_array.reshape(-1, 1), right_eye_right_array.reshape(-1, 1))
    pred = reg.predict(pred_x.reshape(-1, 1))
    ax_right_right.plot(pred_x, pred, color  = "r")    
    coef = reg.coef_
    ax_right_right.text(250,300,"coef = " + convert2str(coef[0]), color  = "r")
    
    reg = LinearRegression().fit(left_Vol_array.reshape(-1, 1), right_eye_left_array.reshape(-1, 1))
    pred = reg.predict(pred_x.reshape(-1, 1))
    ax_right_left.plot(pred_x, pred, color  = "b")    
    coef = reg.coef_
    ax_right_left.text(50,300,"coef = " + convert2str(coef[0]), color  = "b")    
    
    

    '''
    plt.figure()
    plt.scatter(right_Vol_array ,left_eye_right_array, c = "gray")
    plt.plot(right_Vol_array, pred)
    '''
    left_fig.savefig(output_left)
    right_fig.savefig(output_right)

    plt.close('all')
    templete = create_dict_templete()
    ICSPatient = templete["ICSSuiteDBPMRDataSet"]["ICSPatient"]
    ICSPatient["Date"] = user_info[0]
    ICSPatient["Doctor"] = user_info[1]
    ICSPatient["Device"] = user_info[2]
    ICSPatient["Name"] = user_info[3]
    ICSPatient["Exam"] = user_info[4]
    ICSPatient["Mode"] = user_info[5]
    ICSPatient["PatientID"] = date_site_patient_path.split('/')[-1].split('_')[-1]

    HITest = ICSPatient["HITest"]
    HITest["NumAcceptedLeftImpulses"] = str(left_count)
    HITest["NumAcceptedRightImpulses"] = str(right_count)
    HITest["PatientUID"] = mp4_path.split("/")[-1][:-4]
    
    HIImpulse_temp = HITest["HIImpulse"][0].copy()
    HITest["HIImpulse"] = []
    for i in range(len(splitted_data)):
        HIImpulse_temp["Left_EyeVelocitySamples"] = convert2str(left_gradient_list[i])
        HIImpulse_temp["Right_EyeVelocitySamples"] = convert2str(right_gradient_list[i])
        HIImpulse_temp["Gain"] = str((Output_Gain_list[i][0] + Output_Gain_list[i][1])/2)
        HIImpulse_temp["Timestamp"] = str(splitted_data[i][3])
        HeadVel = np.array(splitted_data[i][1])[:,2].astype(np.float32)
        HIImpulse_temp["HeadVelocitySamples"] = convert2str(HeadVel)
        HIImpulse_temp["IsDirectionLeft"] = str(not(splitted_data[i][2]))
        HITest["HIImpulse"].append(HIImpulse_temp.copy())
    
    ICSVideo = ICSPatient["ICSVideo"]
    ICSVideo["FileNameWithPath"] = mp4_path
    ICSVideo["FrameRate"] = str(210)
    ICSVideo["VideoUID"] = mp4_path.split("/")[-1][:-4]



    out_xml = xmltodict.unparse(templete, pretty=True)
    with open(output_xml, 'w') as file:
        file.write(out_xml)
    
    




	










