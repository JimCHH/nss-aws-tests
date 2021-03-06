import pickle

# Test1, 2, 3 general function def below:
def fix_blink(data):
  zero_idx = np.where(data == 0)[0]
  if (len(zero_idx) != 0):
    for i in range(len(zero_idx)):
      if (zero_idx[i] != 0):
        data[zero_idx[i]] = data[zero_idx[i] - 1]
    data_f = data
  else:
    data_f = data

  return data_f

# Test1 (Spontaneous nystagmus test) & Test2 (Gaze evoked test) function def below:
def isoutlier_pks(locs, pks):
  ## Remove noise peak
  c = 1.4826 # c=-1/(sqrt(2)*erfcinv(3/2))
  MAD = c * np.median(abs(pks - np.median(pks)))  # MAD = c*median(abs(A-median(A)))
  outlier_val = [x for x in pks if (x > 3 * MAD)] # ref function in matlab method "median (default)" https://www.mathworks.com/help/matlab/ref/isoutlier.html#bvlllts-method
  tmp1 = []
  for i in range(len(outlier_val)):
    tmp = np.argwhere(pks == outlier_val[i])
    tmp1 = np.append(tmp1, tmp)

  if (tmp1 != []):
    tmp1 = tmp1.astype(int)
    locs_f = np.delete(locs, tmp1)
    pks_f = np.delete(pks, tmp1)
  else:
    locs_f = np.delete(locs, tmp1)
    pks_f = np.delete(pks, tmp1)

  return locs_f, pks_f

def Nystagmus_extract(data, Fs, medfilt1_para):
  ## Load filter parameter
  # Reference to paper: Pander, Tomasz, et al. "1. ." 2012 Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE, 2012.
  FIR1 = np.array(
      [-0.0296451204833518, 0.00925172607229440, -0.0115293989022348, 0.0140375254341020, -0.0167393289436908,
        0.0195876175466524, -0.0225259190063055, 0.0254901124852358, -0.0284104995644265, 0.0312142309050116,
        -0.0338279828852039, 0.0361807618278533, -0.0382067031641653, 0.0398477298031179, -0.0410559384117443,
        0.0417955941208891, 1.00907118633193, 0.0417955941208891, -0.0410559384117443, 0.0398477298031179,
        -0.0382067031641653, 0.0361807618278533, -0.0338279828852039, 0.0312142309050116, -0.0284104995644265,
        0.0254901124852358, -0.0225259190063055, 0.0195876175466524, -0.0167393289436908, 0.0140375254341020,
        -0.0115293989022348, 0.00925172607229440, -0.0296451204833518])
  FIR2 = np.array(
      [0.0126790233155853, 0.00260959042439373, 0.00357784368011279, 0.00457039817392485, 0.00553924417565522,
        0.00642922548646735, 0.00717965385859548, 0.00772634931826725, 0.00800404249455777, 0.00794905231651221,
        0.00750213284050330, 0.00661136771684943, 0.00523498092281557, 0.00334392865357511, 0.000924140148254810,
        -0.00202171512828932, -0.00547304034574246, -0.00939078846528278, -0.0137176607555862, -0.0183790526975593,
        -0.0232847398817529, -0.0283312622978241, -0.0334049321297642, -0.0383853594960468, -0.0431493641058940,
        -0.0475751199533050, -0.0515463660886133, -0.0549565100146286, -0.0577124517959393, -0.0597379665859794,
        -0.0609765005955555, 0.961827736572315, -0.0609765005955555, -0.0597379665859794, -0.0577124517959393,
        -0.0549565100146286, -0.0515463660886133, -0.0475751199533050, -0.0431493641058940, -0.0383853594960468,
        -0.0334049321297642, -0.0283312622978241, -0.0232847398817529, -0.0183790526975593, -0.0137176607555862,
        -0.00939078846528278, -0.00547304034574246, -0.00202171512828932, 0.000924140148254810, 0.00334392865357511,
        0.00523498092281557, 0.00661136771684943, 0.00750213284050330, 0.00794905231651221, 0.00800404249455777,
        0.00772634931826725, 0.00717965385859548, 0.00642922548646735, 0.00553924417565522, 0.00457039817392485,
        0.00357784368011279, 0.00260959042439373, 0.0126790233155853])
  FIR3 = np.array(
      [0.0166161054134519, -0.00210022371807598, 0.00177986913220994, -0.00133019530735843, 0.000740011376541466,
        -1.91201881788006e-17, -0.000896901583959258, 0.00195515458409058, -0.00317631732882165, 0.00455880033204366,
        -0.00609768730834985, 0.00778463016723917, -0.00960782418992972, 0.0115520675165402, -0.0135989068041230,
        0.0157268685236960, -0.0179117729190226, 0.0201271252267365, -0.0223445764326939, 0.0245344436876570,
        -0.0266662785968899, 0.0287094699966587, -0.0306338665908435, 0.0324104039869497, -0.0340117202744467,
        0.0354127443476540, -0.0365912416941331, 0.0375283033368980, -0.0382087650095708, 0.0386215454190648,
        0.930237469421573, 0.0386215454190648, -0.0382087650095708, 0.0375283033368980, -0.0365912416941331,
        0.0354127443476540, -0.0340117202744467, 0.0324104039869497, -0.0306338665908435, 0.0287094699966587,
        -0.0266662785968899, 0.0245344436876570, -0.0223445764326939, 0.0201271252267365, -0.0179117729190226,
        0.0157268685236960, -0.0135989068041230, 0.0115520675165402, -0.00960782418992972, 0.00778463016723917,
        -0.00609768730834985, 0.00455880033204366, -0.00317631732882165, 0.00195515458409058, -0.000896901583959258,
        -1.91201881788006e-17, 0.000740011376541466, -0.00133019530735843, 0.00177986913220994, -0.00210022371807598,
        0.0166161054134519])

  ## Preprocessing stage
  data_m = data - np.mean(data)
  data1 = stats.zscore(data_m)
  data2 = signal.medfilt(data1, medfilt1_para)  # median filter
  # Use lfilter to filter x with the FIR filter.
  data3 = filtfilt(FIR1, 1, data2)  # The low-pass filtering with fcut-off = 30 Hz realized as the 32th order low-pass FIR filter.
  data4 = filtfilt(FIR2, 1, data3)  # The high-pass filtering with fcut-off = 1.5 Hz applying the Chebyshev window with 20 dB of relative sidelobe attenuation is also used. The order of the filter is 62.
  data5 = filtfilt(FIR3, 1, data4)  # The low-pass FIR filtering with fcut-off = 25 Hz realized as the 60th order low-pass FIR filter and the Chebyshev window with 20 dB of relative sidelobe attenuation is also used.

  ## Non-linear operation
  data6 = np.power(np.diff(data5), 2)

  ## Peak detection
  # Nystagmus waveform last as high as 350 ms / mean is 250 ms
  locs, properties = find_peaks(data6, prominence=0.01, distance=Fs*0.1) # distance = 250 / (1000/Fs)
  pks = properties.get('prominences')
  locs_f, pks_f = isoutlier_pks(locs, pks)

  return locs_f, pks_f

def isoutlier(data):
  ## Remove mean outlier
  outlier_val = [x for x in data if (x > 3 * np.std(data))]
  tmp1 = []
  for i in range(len(outlier_val)):
    tmp = np.argwhere(data == outlier_val[i])
    tmp1 = np.append(tmp1, tmp)
  
  if (tmp1 != []):
    tmp1 = tmp1.astype(int)
    data_f = np.delete(data, tmp1)
  else:
    data_f = np.delete(data, tmp1)

  return data_f

def SPV_computation(data, Interval, medfilt1_para):
  ## Slow phase detection
  data_m = data - np.mean(data)
  # true for all elements more than three local scaled MAD from the local median
  c = 1.4826 # c=-1/(sqrt(2)*erfcinv(3/2))
  MAD = c * np.median(abs(np.diff(data_m) - np.median(np.diff(data_m))))  # MAD = c*median(abs(A-median(A)))
  FP_out = np.where(abs(np.diff(data_m)) > (3 * MAD), 0, 1)
  for i in range(1, len(FP_out) - 1):
    if ((FP_out[i-1] & FP_out[i+1]) == 1):
      FP_out[i] = 1
    elif ((FP_out[i-1] | FP_out[i+1]) == 0):
      FP_out[i] = 0
    else:
      FP_out[i] = FP_out[i]
  SP_idx = np.where(FP_out)

  ## Slow Phase Velocity (SPV) parameter
  data_v = np.diff(data_m) / Interval  # for Nystagmus type classification
  SP_v = signal.medfilt(data_v, medfilt1_para) # for SPV computation
  SP_v_SP = SP_v[SP_idx]
  SP_v_SP1 = isoutlier(SP_v_SP) # mean remove outlier
  SPV_mean = np.nanmean(SP_v_SP1)
  SPV_std = np.nanstd(SP_v_SP1)
  SPV_med = np.nanmedian(SP_v_SP1)
  if (SP_v_SP1 != []):
    SPV_iqr = np.subtract(*np.percentile(SP_v_SP1, [75, 25]))
  else:
    SPV_iqr = float("nan")

  ## SPV durartion ratio
  # Every VNG waveform (30sec), the duration of slow phase (right or up) over the duration of show phase (left or down)
  # Modified ratio = (long duration / short duration)???high ratio is with Nystagmus???ratio is 1 without Nystagmus
  SPVd_r = np.sum(np.where(SP_v_SP1 > 0, 1, 0))# * Interval
  SPVd_l = np.sum(np.where(SP_v_SP1 < 0, 1, 0))# * Interval
  if (SPVd_r >= SPVd_l):
    SPVd_ratio = SPVd_r / SPVd_l
  else:
    SPVd_ratio = SPVd_l / SPVd_r
   
  return SPV_mean, SPV_std, SPV_med, SPV_iqr, SPVd_ratio, SP_v, SP_idx, data_m, SP_v_SP, SP_v_SP1
  # data_m: zeromean Eye position
  # SP_v: filtered Eye velocity
  # SP_idx: all slow phase index in Eye position and velocity (green dot)
  # SP_v, SP_v_SP, SP_v_SP1, data_v

def Nystagmus_type(data, locs, axis):
  ## Nystagmus type classification
  # axis = 'Horizontal'
  # axis = 'Vertical'
  data_m = data - np.mean(data)
  data_v = np.diff(data_m) / Interval  # for Nystagmus type classification
  saccade_array = np.sign(data_v[locs])
  saccade_num_P = np.sum(np.where(saccade_array == 1, 1, 0))
  saccade_consecnum_P = max([len(list(g)) for i, g in groupby(saccade_array) if i == 1], default = [])
  saccade_num_N = np.sum(np.where(saccade_array == -1, 1, 0))
  saccade_consecnum_N = max([len(list(g)) for i, g in groupby(saccade_array) if i == -1], default = [])
  saccade_num_Z = np.sum(np.where(saccade_array == 0, 1, 0))
  saccade_consecnum_Z = max([len(list(g)) for i, g in groupby(saccade_array) if i == 0], default = [])
  list1 = [saccade_num_P, saccade_num_N, saccade_num_Z]
  saccade_num_max = list1.index(max(list1))
  if saccade_num_max == 0 and (saccade_num_N/saccade_num_P < 0.2):
    if axis == 'Horizontal':
      nystagmus_type = 'LBN'
    else: # 'Vertical'
      nystagmus_type = 'DBN'
  elif saccade_num_max == 1 and (saccade_num_P/saccade_num_N < 0.2):
    if axis == 'Horizontal':
      nystagmus_type = 'RBN'
    else: # 'Vertical'
      nystagmus_type = 'UBN'
  elif saccade_num_max == 2:
    nystagmus_type = 'Unknown'
  else:
    nystagmus_type = 'Jerks'

  return nystagmus_type

# Test2 (Gaze evoked test) function def below, but the above function need to be defined first:
def ismember(locs, gaze_interval):
  return [ np.sum(a == gaze_interval) for a in locs ]

def gaze_SPV(SP_v, SP_idx):
  SP_v_SP = SP_v[SP_idx]
  SP_v_SP1 = isoutlier(SP_v_SP) # mean remove outlier
  SPV_mean = np.nanmean(SP_v_SP1)
  SPV_std = np.nanstd(SP_v_SP1)
  SPV_med = np.nanmedian(SP_v_SP1)
  if (SP_v_SP1 != []):
    SPV_iqr = np.subtract(*np.percentile(SP_v_SP1, [75, 25]))
  else:
    SPV_iqr = float("nan")

  ## SPV durartion ratio
  # Every VNG waveform (30sec), the duration of slow phase (right or up) over the duration of show phase (left or down)
  # Modified ratio = (long duration / short duration)???high ratio is with Nystagmus???ratio is 1 without Nystagmus
  SPVd_r = np.sum(np.where(SP_v_SP1 > 0, 1, 0))# * Interval
  SPVd_l = np.sum(np.where(SP_v_SP1 < 0, 1, 0))# * Interval
  if (SPVd_r >= SPVd_l):
    SPVd_ratio = SPVd_r / SPVd_l
  else:
    SPVd_ratio = SPVd_l / SPVd_r
  
  return SPV_mean, SPV_std, SPV_med, SPV_iqr, SPVd_ratio, SP_v_SP1

def gaze_interval_split(data_H, data_V):
  # find right left up down interval (1/2*target degree, Horiztonal target degree = 15, Vertical target degree = 10)
  center_interval = np.where((data_H <= 7.5) & (data_H >= -7.5) & (data_V <= 5) & (data_V >= -5)) 
  right_interval = np.where((data_H > 7.5)) # ignore right corner noise
  left_interval = np.where(data_H < -7.5) # ignore leff corner noise
  up_interval = np.where(data_V > 5) # ignore up corner noise
  down_interval = np.where(data_V < -5) # ignore down corner noise

  return center_interval, right_interval, left_interval, up_interval, down_interval

# Test3 (Skew deviation (CUT) test) function def below:
## CUT function skew deviation
# Avg Eye Position Shift (??) ??? the average eye position deviation (for the horizontal and vertical traces) when the condition changes between the eye being covered and uncovered.
def skewD(data):
  skew_deviation = np.subtract(*np.percentile(data, [90, 10]))

  return skew_deviation

### Main code
### Import function
import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
from scipy.signal import filtfilt
from scipy.signal import find_peaks
from itertools import groupby

## System parameter setting
# Predefined video fps
Fs = 210 # 222 for EyeSeeCam
Interval = 1/210 # 222 for EyeSeeCam
medfilt1_para = 11 # filter parameter

# TestN = pkl_list.split('_')[-4] # split string by '_' and output the last 4th string array
def extraction(mp4_path, data):
  TestN = mp4_path[-9:-4]
  T = data['Timestamps'] # load timestamps from data dictionary
  total_time = len(T)/210 # data time (sec)
  saccade_interval = (T[-1] / 210) / 10 # num/10s, T[-1]=total frame
  DATA = {'T': T}
  if TestN == 'Test1' or TestN == 'Test2':
    # DATA is all you need
    DATA['saccade_num'] = {'Left': {}, 'Right': {}}
    DATA['saccade_num_FR'] = {'Left': {}, 'Right': {}}
    DATA['SPV_mean'] = {'Left': {}, 'Right': {}}
    DATA['SPV_std'] = {'Left': {}, 'Right': {}}
    DATA['SPV_med'] = {'Left': {}, 'Right': {}}
    DATA['SPV_iqr'] = {'Left': {}, 'Right': {}}
    DATA['SPVd_ratio'] = {'Left': {}, 'Right': {}}
    DATA['data_m'] = {'Left': {}, 'Right': {}}
    DATA['SP_v'] = {'Left': {}, 'Right': {}}
    DATA['SP_idx'] = {'Left': {}, 'Right': {}}
    DATA['SP_v_SP_outlier_filtered'] = {'Left': {}, 'Right': {}}
    DATA['nystagmus_type'] = {'Left': {}, 'Right': {}}

    ## Horizontal data / Vertial data as input from Left eye / Right eye
    # eye_select = ['Left', 'Right']
    # dir_select = ['Horizontal', 'Vertical']
    for eye in ['Left', 'Right']:
      for axis in ['Horizontal', 'Vertical']:
        ## VNG data fix zero value
        data_f = fix_blink(data[eye][axis])
        
        ## Nystagmus trial detection
        locs, pks = Nystagmus_extract(data_f, Fs, medfilt1_para)
        saccade_num = len(locs)
        saccade_num_FR = saccade_num / saccade_interval
        
        ## SPV parameter computation
        SPV_mean, SPV_std, SPV_med, SPV_iqr, SPVd_ratio, SP_v, SP_idx, data_m, SP_v_SP, SP_v_SP1 = SPV_computation(data_f, Interval, medfilt1_para)

        ## Nystagmus type classification
        nystagmus_type = Nystagmus_type(data_f, locs, axis) # axis use "Horizontal" or "Vertical"

        ## Updata DATA
        DATA['saccade_num'][eye][axis] = saccade_num
        DATA['saccade_num_FR'][eye][axis] = saccade_num_FR
        DATA['SPV_mean'][eye][axis] = SPV_mean
        DATA['SPV_std'][eye][axis] = SPV_std
        DATA['SPV_med'][eye][axis] = SPV_med
        DATA['SPV_iqr'][eye][axis] = SPV_iqr
        DATA['SPVd_ratio'][eye][axis] = SPVd_ratio
        DATA['data_m'][eye][axis] = data_m
        DATA['SP_v'][eye][axis] = SP_v
        DATA['SP_v_SP_outlier_filtered'][eye][axis] = SP_v_SP1
        DATA['SP_idx'][eye][axis] = SP_idx
        DATA['nystagmus_type'][eye][axis] = nystagmus_type

    if TestN == 'Test2':
      ## output all dictionary data
      # Center
      DATA['saccade_num_center'] = {'Left': {}, 'Right': {}}
      DATA['saccade_num_FR_center'] = {'Left': {}, 'Right': {}}
      DATA['SPV_mean_center'] = {'Left': {}, 'Right': {}}
      DATA['SPV_std_center'] = {'Left': {}, 'Right': {}}
      DATA['SPV_med_center'] = {'Left': {}, 'Right': {}}
      DATA['SPV_iqr_center'] = {'Left': {}, 'Right': {}}
      DATA['SPVd_ratio_center'] = {'Left': {}, 'Right': {}}
      DATA['SP_v_SP_outlier_filtered_center'] = {'Left': {}, 'Right': {}}
      # Right
      DATA['saccade_num_right'] = {'Left': {}, 'Right': {}}
      DATA['saccade_num_FR_right'] = {'Left': {}, 'Right': {}}
      DATA['SPV_mean_right'] = {'Left': {}, 'Right': {}}
      DATA['SPV_std_right'] = {'Left': {}, 'Right': {}}
      DATA['SPV_med_right'] = {'Left': {}, 'Right': {}}
      DATA['SPV_iqr_right'] = {'Left': {}, 'Right': {}}
      DATA['SPVd_ratio_right'] = {'Left': {}, 'Right': {}}
      DATA['SP_v_SP_outlier_filtered_right'] = {'Left': {}, 'Right': {}}
      # Left
      DATA['saccade_num_left'] = {'Left': {}, 'Right': {}}
      DATA['saccade_num_FR_left'] = {'Left': {}, 'Right': {}}
      DATA['SPV_mean_left'] = {'Left': {}, 'Right': {}}
      DATA['SPV_std_left'] = {'Left': {}, 'Right': {}}
      DATA['SPV_med_left'] = {'Left': {}, 'Right': {}}
      DATA['SPV_iqr_left'] = {'Left': {}, 'Right': {}}
      DATA['SPVd_ratio_left'] = {'Left': {}, 'Right': {}}
      DATA['SP_v_SP_outlier_filtered_left'] = {'Left': {}, 'Right': {}}
      # Up
      DATA['saccade_num_up'] = {'Left': {}, 'Right': {}}
      DATA['saccade_num_FR_up'] = {'Left': {}, 'Right': {}}
      DATA['SPV_mean_up'] = {'Left': {}, 'Right': {}}
      DATA['SPV_std_up'] = {'Left': {}, 'Right': {}}
      DATA['SPV_med_up'] = {'Left': {}, 'Right': {}}
      DATA['SPV_iqr_up'] = {'Left': {}, 'Right': {}}
      DATA['SPVd_ratio_up'] = {'Left': {}, 'Right': {}}
      DATA['SP_v_SP_outlier_filtered_up'] = {'Left': {}, 'Right': {}}
      # Down
      DATA['saccade_num_down'] = {'Left': {}, 'Right': {}}
      DATA['saccade_num_FR_down'] = {'Left': {}, 'Right': {}}
      DATA['SPV_mean_down'] = {'Left': {}, 'Right': {}}
      DATA['SPV_std_down'] = {'Left': {}, 'Right': {}}
      DATA['SPV_med_down'] = {'Left': {}, 'Right': {}}
      DATA['SPV_iqr_down'] = {'Left': {}, 'Right': {}}
      DATA['SPVd_ratio_down'] = {'Left': {}, 'Right': {}}
      DATA['SP_v_SP_outlier_filtered_down'] = {'Left': {}, 'Right': {}}


      ## Horizontal data / Vertial data as input from Left eye / Right eye
      # eye_select = ['Left', 'Right']
      # dir_select = ['Horizontal', 'Vertical']
      for eye in ['Left', 'Right']:
        ## VNG data fix zero value
        data_f0 = fix_blink(data[eye]['Horizontal'])
        data_f1 = fix_blink(data[eye]['Vertical'])

        center_interval, right_interval, left_interval, up_interval, down_interval = gaze_interval_split(data_f0, data_f1)
        SP_idx_center = np.intersect1d(SP_idx, center_interval)
        SP_idx_right = np.intersect1d(SP_idx, right_interval)
        SP_idx_left = np.intersect1d(SP_idx, left_interval)
        SP_idx_up = np.intersect1d(SP_idx, up_interval)
        SP_idx_down = np.intersect1d(SP_idx, down_interval)

        for axis in ['Horizontal', 'Vertical']:
          ## VNG data fix zero value
          data_f = fix_blink(data[eye][axis])

          ## Nystagmus trial detection
          locs, pks = Nystagmus_extract(data_f, Fs, medfilt1_para)

          ## SPV parameter computation
          SPV_mean, SPV_std, SPV_med, SPV_iqr, SPVd_ratio, SP_v, SP_idx, data_m, SP_v_SP, SP_v_SP1 = SPV_computation(data_f, Interval, medfilt1_para)
          
          # Gaze SPV parameter computation
          SPV_mean_center, SPV_std_center, SPV_med_center, SPV_iqr_center, SPVd_ratio_center, SP_v_SP1_center = gaze_SPV(SP_v, SP_idx_center)
          SPV_mean_right, SPV_std_right, SPV_med_right, SPV_iqr_right, SPVd_ratio_right, SP_v_SP1_right = gaze_SPV(SP_v, SP_idx_right)
          SPV_mean_left, SPV_std_left, SPV_med_left, SPV_iqr_left, SPVd_ratio_left, SP_v_SP1_left = gaze_SPV(SP_v, SP_idx_left)
          SPV_mean_up, SPV_std_up, SPV_med_up, SPV_iqr_up, SPVd_ratio_up, SP_v_SP1_up = gaze_SPV(SP_v, SP_idx_up)
          SPV_mean_down, SPV_std_down, SPV_med_down, SPV_iqr_down, SPVd_ratio_down, SP_v_SP1_down = gaze_SPV(SP_v, SP_idx_down)

          # Gaze Nystagmus trial detection
          saccade_num_center = np.sum(ismember(locs, SP_idx_center))
          saccade_num_right = np.sum(ismember(locs, SP_idx_right))
          saccade_num_left = np.sum(ismember(locs, SP_idx_left))
          saccade_num_up = np.sum(ismember(locs, SP_idx_up))
          saccade_num_down = np.sum(ismember(locs, SP_idx_down))
          saccade_num_FR_center = saccade_num_center / saccade_interval
          saccade_num_FR_right = saccade_num_right / saccade_interval
          saccade_num_FR_left = saccade_num_left / saccade_interval
          saccade_num_FR_up = saccade_num_up / saccade_interval
          saccade_num_FR_down = saccade_num_down / saccade_interval
        
          ## Update dictionary data
          # Center
          DATA['saccade_num_center'][eye][axis] = saccade_num_center
          DATA['saccade_num_FR_center'][eye][axis] = saccade_num_FR_center
          DATA['SPV_mean_center'][eye][axis] = SPV_mean_center
          DATA['SPV_std_center'][eye][axis] = SPV_std_center
          DATA['SPV_med_center'][eye][axis] = SPV_med_center
          DATA['SPV_iqr_center'][eye][axis] = SPV_iqr_center
          DATA['SPVd_ratio_center'][eye][axis] = SPVd_ratio_center
          DATA['SP_v_SP_outlier_filtered_center'][eye][axis] = SP_v_SP1_center
          # Right
          DATA['saccade_num_right'][eye][axis] = saccade_num_right
          DATA['saccade_num_FR_right'][eye][axis] = saccade_num_FR_right
          DATA['SPV_mean_right'][eye][axis] = SPV_mean_right
          DATA['SPV_std_right'][eye][axis] = SPV_std_right
          DATA['SPV_med_right'][eye][axis] = SPV_med_right
          DATA['SPV_iqr_right'][eye][axis] = SPV_iqr_right
          DATA['SPVd_ratio_right'][eye][axis] = SPVd_ratio_right
          DATA['SP_v_SP_outlier_filtered_right'][eye][axis] = SP_v_SP1_right
          # Left
          DATA['saccade_num_left'][eye][axis] = saccade_num_left
          DATA['saccade_num_FR_left'][eye][axis] = saccade_num_FR_left
          DATA['SPV_mean_left'][eye][axis] = SPV_mean_left
          DATA['SPV_std_left'][eye][axis] = SPV_std_left
          DATA['SPV_med_left'][eye][axis] = SPV_med_left
          DATA['SPV_iqr_left'][eye][axis] = SPV_iqr_left
          DATA['SPVd_ratio_left'][eye][axis] = SPVd_ratio_left
          DATA['SP_v_SP_outlier_filtered_left'][eye][axis] = SP_v_SP1_left
          # Up
          DATA['saccade_num_up'][eye][axis] = saccade_num_up
          DATA['saccade_num_FR_up'][eye][axis] = saccade_num_FR_up
          DATA['SPV_mean_up'][eye][axis] = SPV_mean_up
          DATA['SPV_std_up'][eye][axis] = SPV_std_up
          DATA['SPV_med_up'][eye][axis] = SPV_med_up
          DATA['SPV_iqr_up'][eye][axis] = SPV_iqr_up
          DATA['SPVd_ratio_up'][eye][axis] = SPVd_ratio_up
          DATA['SP_v_SP_outlier_filtered_up'][eye][axis] = SP_v_SP1_up
          # Down
          DATA['saccade_num_down'][eye][axis] = saccade_num_down
          DATA['saccade_num_FR_down'][eye][axis] = saccade_num_FR_down
          DATA['SPV_mean_down'][eye][axis] = SPV_mean_down
          DATA['SPV_std_down'][eye][axis] = SPV_std_down
          DATA['SPV_med_down'][eye][axis] = SPV_med_down
          DATA['SPV_iqr_down'][eye][axis] = SPV_iqr_down
          DATA['SPVd_ratio_down'][eye][axis] = SPVd_ratio_down
          DATA['SP_v_SP_outlier_filtered_down'][eye][axis] = SP_v_SP1_down

      DATA['center_interval'] = center_interval
      DATA['right_interval'] = right_interval
      DATA['left_interval'] = left_interval
      DATA['up_interval'] = up_interval
      DATA['down_interval'] = down_interval

  elif TestN == 'Test3':
    # if TestN == 'Test3'
    ## output all dictionary data
    DATA['data_m'] = {'Left': {}, 'Right': {}}
    DATA['skew_deviation'] = {'Left': {}, 'Right': {}}

    ## Horizontal data / Vertial data as input from Left eye / Right eye
    # eye_select = ['Left', 'Right']
    # dir_select = ['Horizontal', 'Vertical']
    for eye in ['Left', 'Right']:
      for axis in ['Horizontal', 'Vertical']:
        ## VNG data fix zero value
        data_f = fix_blink(data[eye][axis])
        
        ## Preprocessing stage for zero mean
        data_m = data_f - np.mean(data_f)

        ## Compute Skew deviation for angle diffrence
        skew_deviation = skewD(data_m)

        ## Updata dictionary data
        DATA['data_m'][eye][axis] = data_m
        DATA['skew_deviation'][eye][axis] = skew_deviation

  # DATA save as pkl for Test1 or Test2 or Test3
  with open(mp4_path[:-4] + '_sp_dataset_API.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
      pickle.dump(DATA, f)