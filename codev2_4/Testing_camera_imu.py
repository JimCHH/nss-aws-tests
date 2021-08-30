import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import serial
import matplotlib.ticker as ticker
from math import sin, cos, pi
import os
import csv


# import datetime
# import matplotlib.animation as animation


def dataConv(data):
    # rawdata轉換
    acc_sensitivity = 2048
    # acc_sensitivity = 1
    gyro_sensitivity = 16.4
    try:
        ACC_X = int(data[0]) * 256 + int(data[1])
        ACC_Y = int(data[2]) * 256 + int(data[3])
        ACC_Z = int(data[4]) * 256 + int(data[5])
        GYR_X = int(data[6]) * 256 + int(data[7])
        GYR_Y = int(data[8]) * 256 + int(data[9])
        GYR_Z = int(data[10]) * 256 + int(data[11])
    except IndexError:
        pass
    except ValueError:
        pass
    except UnboundLocalError:
        pass
    #    MAG_X = int(data[12]) * 256 + int(data[13])
    #    MAG_Y = int(data[14]) * 256 + int(data[15])
    #    MAG_Z = int(data[16]) * 256 + int(data[17])
    # print("{")
    # print("ACC_X : " + str(ACC_X))
    # print("ACC_Y : " + str(ACC_Y))
    # print("ACC_Z : " + str(ACC_Z))
    # print("GYR_X : " + str(GYR_X))
    # print("GYR_Y : " + str(GYR_Y))
    # print("GYR_Z : " + str(GYR_Z))
    # print("MAG_X : " + str(MAG_X))
    # print("MAG_Y : " + str(MAG_Y))
    # print("MAG_Z : " + str(MAG_Z))
    # print("}")
    #    temp = [ACC_X, ACC_Y, ACC_Z,
    #            GYR_X, GYR_Y, GYR_Z,
    #            MAG_X, MAG_Y, MAG_Z]

    try:
        temp = [ACC_X, ACC_Y, ACC_Z,
                GYR_X, GYR_Y, GYR_Z]
    except UnboundLocalError:
        temp = [0, 0, 0, 0, 0, 0]
        pass
    for i in range(len(temp)):
        if temp[i] > 32768:
            temp[i] = temp[i] - 65536
        else:
            temp[i] = temp[i]
    temp[0] = temp[0] / acc_sensitivity
    temp[1] = temp[1] / acc_sensitivity
    temp[2] = temp[2] / acc_sensitivity
    temp[3] = temp[3] / gyro_sensitivity
    temp[4] = temp[4] / gyro_sensitivity
    temp[5] = temp[5] / gyro_sensitivity

    return temp


def getTime():
    now = time.localtime()
    current_time = str(now.tm_mon) + '' + str(now.tm_mday) + '_' + \
                   str(now.tm_hour) + '_' + str(now.tm_min) + '_' + str(now.tm_sec)
    return current_time


def CollectAndShowIMU(time_):
    use_imu = True
    dps = 70
    t = 1 / dps
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 400)
    cap.set(5, 210)
    cap.get(cv2.CAP_PROP_BITRATE)
    ser = serial.Serial('COM3', 115200, bytesize=8, parity='N', stopbits=1)
    start_time = time.time()
    # global IMU_GYRO
    IMU_GYRO = []
    # plt.show()
    total_time = 0
    time1 = 0
    lastAngelY = 0
    frames = []
    current_time = getTime()
    path = './video/'
    mkdir(path)
    save_name = current_time + '.avi'
    filename = path + save_name
    fps = 60
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (1280, 400))

    while total_time < int(time_):

        current_time = time.time()
        sline = ser.readline()
        total_time = current_time - start_time
        out10 = dataConv(sline.decode("utf-8").split(","))
        # print(out10)

        ret, frame = cap.read()
        frames.append(frame)
        video_writer.write(frame)

        AngleY = lastAngelY + out10[4] * t
        IMU_GYRO.append([out10[3], out10[4], out10[5], AngleY])

        lastAngelY = AngleY
        # print("RET: "+str(ret))
        cv2.putText(frame, "Angle: " + str(int(AngleY)), (30, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if total_time - time1 > 1:
            print(int(time_) - round(total_time))
            time1 = total_time

        # print(total_time)
        # save.append([out10[3], out10[4], out10[5], time1 - start])

        # if time.time() - temp_time > 1 / 70:
        #     ax1.clear()
        #     ax2.clear()
        #     ax3.clear()
        #
        #     plot_data = np.array(save)
        #
        #     ax1.plot(plot_data[:, 3], plot_data[:, 0])
        #     ax2.plot(plot_data[:, 3], plot_data[:, 1])
        #     ax3.plot(plot_data[:, 3], plot_data[:, 2])
        #
        #     ax1.set_xlim(0, 30)
        #     ax1.set_ylim(-200, 200)
        #
        #     ax2.set_xlim(0, 30)
        #     ax2.set_ylim(-200, 200)
        #
        #     ax3.set_xlim(0, 30)
        #     ax3.set_ylim(-300, 300)
        #
        #     plt.pause(0.01)
        #     temp_time = time.time()
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print('0')
    print('done')
    return IMU_GYRO, filename, frames


def getGyro(conv_data):
    gyro_x = []
    gyro_y = []
    gyro_z = []
    for data in conv_data:
        temp = data[0]
        gyro_x.append(temp)
        temp = data[1]
        gyro_y.append(temp)
        temp = data[2]
        gyro_z.append(temp)
    return [gyro_x, gyro_y, gyro_z]


def pltGyro(gx, gy, gz):
    dpi = 92
    plt.figure(figsize=(800 / dpi, 800 / dpi), dpi=dpi)
    t = np.arange(0, len(gx) / 70, 1 / 70)
    t = t_correct(t, gx)
    plt.subplot(3, 1, 1)
    plt.plot(t, gx, color='red')
    plt.ylabel("angelSpeed")
    plt.title("X")
    plt.text(t[np.argmax(gx)], y=max(gx), s="MAX:" + str(round(max(gx), ndigits=3)))
    plt.text(t[np.argmin(gx)], y=min(gx), s="MIN:" + str(round(min(gx), ndigits=3)))

    plt.subplot(3, 1, 2)
    plt.plot(t, gy, color='green')
    plt.ylabel("angelSpeed")
    plt.title("Y")
    plt.text(t[np.argmax(gy)], y=max(gy), s="MAX:" + str(round(max(gy), ndigits=3)))
    plt.text(t[np.argmin(gy)], y=min(gy), s="MIN:" + str(round(min(gy), ndigits=3)))

    plt.subplot(3, 1, 3)
    plt.plot(t, gz, color='blue')
    plt.xlabel("time")
    plt.ylabel("angelSpeed")
    plt.title("Z")
    plt.text(t[np.argmax(gz)], y=max(gz), s="MAX:" + str(round(max(gz), ndigits=3)))
    plt.text(t[np.argmin(gz)], y=min(gz), s="MIN:" + str(round(min(gz), ndigits=3)))
    return


def t_correct(t, a):
    if len(t) != len(a):
        t = t[0:-1]
    else:
        return t
    return t


def spread2three(conv_data):
    acc_x = []
    acc_y = []
    acc_z = []
    for data in conv_data:
        temp = data[0]
        acc_x.append(temp)
        temp = data[1]
        acc_y.append(temp)
        temp = data[2]
        acc_z.append(temp)
    return [acc_x, acc_y, acc_z]


def three_into_one_list(a, b, c):
    onelist = []
    for i in range(0, len(a)):
        temp = [a[i], b[i], c[i]]
        onelist.append(temp)
    return onelist


def calcAngel_acc(gyro_x, gyro_y, gyro_z):
    data_num = len(gyro_x)
    dps = 70
    t = 1 / dps
    gyro_ax = []
    gyro_ay = []
    gyro_az = []
    gyro_ax.append(gyro_x[0] / t)
    gyro_ay.append(gyro_y[0] / t)
    gyro_az.append(gyro_z[0] / t)

    for i in range(1, len(gyro_x) - 1):
        gyro_ax.append((gyro_x[i + 1] - gyro_x[i]) / t)
    for i in range(1, len(gyro_y) - 1):
        gyro_ay.append((gyro_y[i + 1] - gyro_y[i]) / t)
    for i in range(1, len(gyro_z) - 1):
        gyro_az.append((gyro_z[i + 1] - gyro_z[i]) / t)
    return [gyro_ax, gyro_ay, gyro_az]


def calcAngel(gyro_data):
    data_num = len(gyro_data)
    dps = 70
    # total_time = data_num / dps
    # t = total_time / data_num
    t = 1 / dps

    temp = []
    for j in range(0, 3):
        angle = [0]
        for i in range(1, data_num):
            angle.append(angle[i - 1] + gyro_data[i - 1][j] * t)
        temp.append(angle)

    angle_point = []
    for i in range(0, data_num):
        temp1 = [temp[0][i], temp[1][i], temp[2][i]]
        angle_point.append(temp1)
    return angle_point


def pltAngle(ag_x, ag_y, ag_z):
    dpi = 92
    plt.figure(figsize=(800 / dpi, 800 / dpi), dpi=dpi)

    t = np.arange(0, len(ag_x) / 70, 1 / 70)
    t = t_correct(t, ag_x)
    plt.subplot(3, 1, 1)
    plt.plot(t, ag_x, color='red')
    plt.ylabel("angle")
    plt.title("X")
    plt.text(t[np.argmax(ag_x)], y=max(ag_x), s="MAX:" + str(round(max(ag_x), ndigits=3)))
    plt.text(t[np.argmin(ag_x)], y=min(ag_x), s="MIN:" + str(round(min(ag_x), ndigits=3)))

    plt.subplot(3, 1, 2)
    plt.plot(t, ag_y, color='green')
    plt.ylabel("angle")
    plt.title("Y")
    plt.text(t[np.argmax(ag_y)], y=max(ag_y), s="MAX:" + str(round(max(ag_y), ndigits=3)))
    plt.text(t[np.argmin(ag_y)], y=min(ag_y), s="MIN:" + str(round(min(ag_y), ndigits=3)))

    plt.subplot(3, 1, 3)
    plt.plot(t, ag_z, color='blue')
    plt.xlabel("time")
    plt.ylabel("angle")
    plt.title("Z")
    plt.text(t[np.argmax(ag_z)], y=max(ag_z), s="MAX:" + str(round(max(ag_z), ndigits=3)))
    plt.text(t[np.argmin(ag_z)], y=min(ag_z), s="MIN:" + str(round(min(ag_z), ndigits=3)))

    plt.tight_layout()

    # plt.pi
    # plt.show()
    return


def pltAngleOnlyY(ag_y):
    dpi = 92
    # plt.figure(figsize=(800 / dpi, 800 / dpi), dpi=dpi)
    t = np.arange(0, len(ag_y) / 70, 1 / 70)
    t = t_correct(t, ag_y)
    t1 = []

    maxPoint_ty = t[np.argmax(ag_y)]
    minPoint_ty = t[np.argmin(ag_y)]

    maxPointY_index = np.argmax(ag_y)
    maxPointY_index_m5 = np.argmax(ag_y) - 5

    minPointY_index = np.argmin(ag_y)
    minPointY_index_m5 = np.argmin(ag_y) - 5
    if abs(max(ag_y)) > abs(min(ag_y)):
        maxPoint_ty = maxPoint_ty
        while ag_y[maxPointY_index] - ag_y[maxPointY_index_m5] < 5:
            maxPoint_ty = t[maxPointY_index_m5]
            maxPointY_index = maxPointY_index_m5
            maxPointY_index_m5 = maxPointY_index_m5 - 5
    else:
        maxPoint_ty = minPoint_ty
        ag_y = [x * (-1) for x in ag_y]
        while ag_y[minPointY_index] - ag_y[minPointY_index_m5] < 5:
            maxPoint_ty = t[minPointY_index_m5]
            minPointY_index = minPointY_index_m5
            minPointY_index_m5 = minPointY_index_m5 - 5

    for i in range(0, len(t)):
        t1.append(int(round(t[i] - (maxPoint_ty - 0.1), 1) * 1000))
    fig, ax = plt.subplots(1, 1)

    plt.ylabel("angle")
    plt.title("Y")
    maxPoint_ty_ind = np.where(t == maxPoint_ty)
    maxPoint_ty_ind = int(maxPoint_ty_ind[0])
    plt.text(t[np.argmax(ag_y)], y=max(ag_y), s="MAX:" + str(round(max(ag_y), ndigits=3)))
    # plt.text(maxPoint_ty, y=max(ag_y), s="MAX:" + str(round(ag_y[maxPoint_ty_ind], ndigits=3)))
    # plt.text(t[np.argmin(ag_y)], y=min(ag_y), s="MIN:" + str(round(min(ag_y), ndigits=3)))
    plt.plot(t, ag_y, color='green')
    plt.xticks(t, t1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.xlim(maxPoint_ty - 0.2, maxPoint_ty + 0.5)
    plt.xlabel('time(ms)')

    # plt.tight_layout()

    # plt.pi
    # plt.show()
    return


def pltGyroOnlyY(gy):
    dpi = 92
    # plt.figure(figsize=(800 / dpi, 800 / dpi), dpi=dpi)
    t = np.arange(0, len(gy) / 70, 1 / 70)
    t = t_correct(t, gy)
    t1 = []
    maxPoint_ty = t[np.argmax(gy)]
    minPoint_ty = t[np.argmin(gy)]
    if abs(max(gy)) > abs(min(gy)):
        maxPoint_ty = maxPoint_ty
        isLeft = True
    else:
        maxPoint_ty = minPoint_ty
        gy = [x * (-1) for x in gy]
        isLeft = False
    for i in range(0, len(t)):
        t1.append(int(round(t[i] - (maxPoint_ty - 0.1), 1) * 1000))

    fig, ax = plt.subplots(1, 1)

    plt.ylabel("angelSpeed")
    plt.title("Y")
    plt.text(t[np.argmax(gy)], y=max(gy), s="MAX:" + str(round(max(gy), ndigits=3)))
    # plt.text(t[np.argmin(gy)], y=min(gy), s="MIN:" + str(round(min(gy), ndigits=3)))
    plt.plot(t, gy, color='green')
    plt.xticks(t, t1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.xlim(maxPoint_ty - 0.2, maxPoint_ty + 0.5)
    plt.xlabel('time(ms)')
    # plt.tight_layout()
    return


def pltGyroOnlyY2(gy):
    # if abs(max(gy)) > abs(min(gy)):
    #     maxY = abs(max(gy))
    #     isLeft = True
    # else:
    #     maxY = abs(min(gy))
    #     gy = [x * (-1) for x in gy]
    #     isLeft = False
    # print(maxY)
    t = np.arange(0, len(gy) / 70, 1 / 70)
    t = t_correct(t, gy)
    tMAX = t[np.argmax(gy)]
    t3 = np.arange(tMAX - 0.2, tMAX + 0.5, 1 / 70)
    indext = np.where(t == t3[0])
    t3_In_t_index_start = int(indext[0])
    t3_In_t_index_end = t3_In_t_index_start + len(t3) - 1
    new_gy = gy[t3_In_t_index_start:t3_In_t_index_end + 1]
    a = round(round(t[np.argmax(gy)], 1) - 0.1, 1)
    fig, ax = plt.subplots(1, 1)
    t3 = [(x - a) * 1000 for x in t3]
    plt.plot(t3, new_gy, color='green')
    return


def pltAngleOnlyY2(gy):
    # if abs(max(gy)) > abs(min(gy)):
    #     maxY = abs(max(gy))
    #     isLeft = True
    # else:
    #     maxY = abs(min(gy))
    #     gy = [x * (-1) for x in gy]
    #     isLeft = False
    # print(maxY)
    t = np.arange(0, len(gy) / 70, 1 / 70)
    t = t_correct(t, gy)
    tMAX = t[np.argmax(gy)]
    # t = [x * 1000 for x in t]
    t3 = np.arange(tMAX - 0.2, tMAX + 0.5, 1 / 70)
    new_t3 = np.arange(-0.1, 0.6 + 1 / 70, 1 / 70)
    print(len(new_t3))
    indext = np.where(t == t3[0])
    print(t3[0])
    t3_In_t_index_start = int(indext[0])
    t3_In_t_index_end = t3_In_t_index_start + len(t3) - 1
    new_gy = gy[t3_In_t_index_start:t3_In_t_index_end + 1]
    a = round(round(t[np.argmax(gy)], 1) - 0.1, 1)
    # a = t[np.argmax(gy)]
    print(a)
    fig, ax = plt.subplots(1, 1)
    t3 = [(x - a) * 1000 for x in t3]
    # t3 = [x * 1000 for x in t3]
    plt.plot(t3, new_gy, color='green')
    # new_t3 = [x * 1000 for x in new_t3]
    # plt.xticks(t3,new_t3)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # plt.show()
    return


def RotationMatrix_X330(x):
    # print(x)
    a = []
    Arc = (-30 / 360) * pi
    RMatrix = np.mat([
        [1, 0, 0],
        [0, cos(Arc), -sin(Arc)],
        [0, sin(Arc), cos(Arc)]
    ])
    for data in x:
        data = np.mat(data)
        y = data.dot(RMatrix)
        y = y.tolist()
        a.append(y[0])

    return a


def saveVideo(frames, save_name):
    fps = 60
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    path = './video/'
    mkdir(path)
    save_name = path + save_name
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, (1280, 400))
    for frame in frames:
        # cv2.imshow('0', frame)
        key = cv2.waitKey(1)
        cv2.imshow('0', frame)
        if key == ord('q') or key == 27:  # Esc
            print('break')
            break
        video_writer.write(frame)

    video_writer.release()
    cv2.destroyAllWindows()
    return


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    return


def delFiles(fileName):
    try:
        os.remove(fileName)
    except OSError as e:
        print(e)
    return


def saveIMU(IMU, fileName):
    path = './IMU_Data/'
    mkdir(path)
    s = fileName.split('/')
    a = s[-1].replace('.avi', '')
    filename = path + a + '.csv'
    with open(filename, 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(IMU)

    return
1

# def calcCurrentAngle(last_gy, gy, time):
#     ini_gy = 0
#     gy = la
#     return


# Collect events until released

# def update(i):
#     sline = ser.readline()
#     time1 = time.time()
#     out10 = dataConv(sline.decode("utf-8").split(","))
#     print(out10[4])
#     line.set_ydata(out10[4])  # update the data.
#     line.set_xdata(time1 - start)
#     line.set_marker("*")
#     return line,


# use_imu = True
#
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(3, 1280)
# cap.set(4, 400)
# cap.set(5, 210)
# cap.get(cv2.CAP_PROP_BITRATE)
# Define the codec and create VideoWriter object
# fourcc = cv2.cv.CV_FOURCC(*'DIVX')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
# out = cv2.VideoWriter('test.mp4', -1, 210.0, (1280, 400))
counter = 0
start = time.time()

"""
fig, ax = plt.subplots()
line, = ax.plot(0, 0, "*")

def init():
    line, = ax.plot(0, 0, "*")
    ax.set_xlim(0,20)
    ax.set_ylim(-200,200)
"""

total_data = []

s = ''
print("頭部操作規範"
      "\n角度:10~20"
      "\n角速度:120~250"
      "\n角加速度:1200~2500")
while not s.isdigit():
    s = input('請輸入秒數(ENTER後開始): ')
    if not s.isdigit():
        print("非數字")

IsContinue = 'y'
degree = []
while IsContinue == 'y':
    print("開始錄製")
    IMU_GYRO, filename, frames = CollectAndShowIMU(s)
    # fig = plt.figure()
    # ax1 = plt.subplot(3, 1, 1)
    # ax2 = plt.subplot(3, 1, 2)
    # ax3 = plt.subplot(3, 1, 3)
    [gyro_x, gyro_y, gyro_z] = getGyro(IMU_GYRO)
    onelist_gyro = three_into_one_list(gyro_x, gyro_y, gyro_z)
    onelist_gyro = RotationMatrix_X330(onelist_gyro)
    [gyro_x, gyro_y, gyro_z] = spread2three(onelist_gyro)
    Angle = calcAngel(onelist_gyro)
    [gyro_ax, gyro_ay, gyro_az] = calcAngel_acc(gyro_x, gyro_y, gyro_z)
    [angle_x, angle_y, angle_z] = spread2three(Angle)
    print('\n---------------------------------------------------')

    maxAngel = int(max([abs(x) for x in angle_y]))
    maxAngelSpeed = int(max([abs(x) for x in gyro_y]))
    maxAngelAcc = int(max([abs(x) for x in gyro_ay]))
    if 250 >= maxAngelSpeed >= 120 and \
            20 >= maxAngel >= 10 and \
            2500 >= maxAngelAcc >= 1200:
        print("頭部操作符合規範")
        # saveIMU(IMU_GYRO, filename)
    else:
        print("頭部操作不符合規範")
        saveIMU(IMU_GYRO, filename)
        # delFiles(filename)

    print("最大角度: " + str(maxAngel))
    print("最大角速度: " + str(maxAngelSpeed))
    print("最大角加速度: " + str(maxAngelAcc))
    degree.append(maxAngel)
    pltGyroOnlyY(gyro_y)
    pltAngleOnlyY(angle_y)

    # pltGyroOnlyY2(angle_y)
    # pltGyro(gyro_x, gyro_y, gyro_z)
    # pltAngle(angle_x, angle_y, angle_z)
    print("關閉圖像以繼續下一筆...")
    plt.show()
    IsContinue = input("繼續下一筆嗎?(y/n): ")

# savePath = './maxDegree/'
# filename = '0505_0845_noRM'
# with open(savePath + 'convertedData_' + filename + '.csv',
#           'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(degree)
print("結束")

# ani = animation.FuncAnimation(fig, update, frames = 500, interval = 1, init_func=init, blit=False)

'''
while (cap.isOpened()):
    if counter == 0:
        t = time.time()
    ret, frame = cap.read()

    if use_imu:
        if ser == []:
            ser = serial.Serial('COM3', 115200, bytesize=8, parity='N', stopbits=1)
    if ret == True:

        if use_imu:
            sline = ser.readline()
            time1 = time.time()
            out10 = dataConv(sline.decode("utf-8").split(","))
        #            ax.plot(start-time1, out10[4],'*')
        # write the flipped frame
        out.write(frame)
        counter += 1

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            total = time.time() - t
            break
        if counter == 1000:
            total = time.time() - t

            break


    else:
        break

# Release everything if job is finished
cap.release()
out.release()

cv2.destroyAllWindows()
'''
# fps = counter / total
# print(fps)
