# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 06:14:11 2021

@author: lab70929
"""
import glob
import cv2
import os




data_list = glob.glob("test_dataset/*.avi")
num = 50
count = 0
for data_name in data_list:
    
    folder_name = str(data_name[:-4])
    try:
        os.mkdir(folder_name)
    except:
        print("資料夾已存在, name = " + folder_name)    
    cap = cv2.VideoCapture(data_name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
    


    for i in range(num):    
        index = int(i*(total_frames/num))
        cap.set(1,index)
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(folder_name+'/'+str(index)+".jpg", gray)
    
    
    
        
    
    