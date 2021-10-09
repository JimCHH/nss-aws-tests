from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, column, gridplot, layout
from bokeh.io import output_notebook, show, export_svg, export_svgs, curdoc, output_file
from bokeh.io.export import get_svg, get_svgs
from bokeh.themes import Theme
from bokeh.models import Div, plots, LinearColorMapper, CDSView, ColumnDataSource, IndexFilter, GroupFilter
#from bokeh.palettes import Spectral3, Spectral4
from bokeh.transform import jitter, linear_cmap
from varname import argname2
import pickle
import math
import pandas as pd
#import colorcet as cc
from scipy.interpolate import CubicSpline
import numpy as np
from datetime import datetime
import re
import time
import calendar
import os
from google.colab import drive 
from selenium import webdriver

# import reportlab library
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import Table, TableStyle, BaseDocTemplate, NextPageTemplate, Frame, PageTemplate, PageBreak
from reportlab.lib import colors
from reportlab.lib.units import mm
from svglib.svglib import svg2rlg

#------------------------------------------------------------------------------
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage') 
wd = webdriver.Chrome('chromedriver',chrome_options=chrome_options)
 
print(wd.current_url)

#------------------------------------------------------------------------------
def nested_dict_to_pd_df(dict_, as_idx = False):
  # load dict name as col name
  var_name = argname2('dict_').replace('_dict','')
  
  var_name = 'velocity' if (var_name == 'SP_v') else var_name
  var_name = 'position' if (var_name == 'data_m') else var_name

  if (as_idx is False):
    # load dict
    df_LH = pd.DataFrame(dict_['Left']['Horizontal'], columns=[var_name])
    df_LV = pd.DataFrame(dict_['Left']['Vertical'], columns=[var_name])
    df_RH = pd.DataFrame(dict_['Right']['Horizontal'], columns=[var_name])
    df_RV = pd.DataFrame(dict_['Right']['Vertical'], columns=[var_name])
    df_T = pd.DataFrame(T, columns=[var_name])
    df = pd.concat([df_T, df_LH.reset_index(drop=True), df_LV.reset_index(drop=True),
            df_RH.reset_index(drop=True), df_RV.reset_index(drop=True)], 
            axis=1)
    df.columns = pd.MultiIndex.from_tuples(zip(['Time',var_name, var_name, var_name, var_name],
                                               ['Time','OS', 'OS', 'OD', 'OD'], 
                                             ['Time','Vertical', 'Horizontal', 'Vertical', 'Horizontal']))
  else:
    df_LH = pd.DataFrame(dict_['Left']['Horizontal']).T
    df_LV = pd.DataFrame(dict_['Left']['Vertical']).T
    df_RH = pd.DataFrame(dict_['Right']['Horizontal']).T
    df_RV = pd.DataFrame(dict_['Right']['Vertical']).T
    df = pd.concat([df_LH.reset_index(drop=True), df_LV.reset_index(drop=True),
              df_RH.reset_index(drop=True), df_RV.reset_index(drop=True)], 
              axis=1)
    df.columns = pd.MultiIndex.from_tuples(zip([var_name, var_name, var_name, var_name],
                                               ['OS', 'OS', 'OD', 'OD'], 
                                              ['Vertical', 'Horizontal', 'Vertical', 'Horizontal']))
    df= df.fillna(-1).astype(int)


  return df, var_name #.unstack(level=1)

#------------------------------------------------------------------------------
# this part will be used to call theme colors
line_color_palatte = {'greens':["#53EE5D", "#14EE22", '#00B30C','#007B08'],
                      'oranges': ["#FFE359", "#FFD815",'#E3BD00','#9D8300']}

#------------------------------------------------------------------------------
def get_spv_scatter_stat_by_group(data):
  GROUP = ['OS Horizontal','OD Horizontal','OS Vertical','OD Vertical']

  source = ColumnDataSource(data)
  plot_size_and_tools = {'height': 400, 'width': 400,
                        'tools':['box_select', 'reset', 'help'],
                        'y_axis_label': 'Velocity (°/s)',
                        'min_border_right':20,
                        'toolbar_location': None,
                        'output_backend':'svg',
                        'match_aspect': True} 
  p = figure(x_range=GROUP, 
            title="SPV Analysis",sizing_mode = 'fixed', **plot_size_and_tools)

  p.scatter(x=0.5, y=jitter('SP_v_SP_outlier_filtered_OS_Horizontal',
                     width=1, range=p.y_range),  source=source, alpha=0.3)
  p.scatter(x=2.5, y=jitter('SP_v_SP_outlier_filtered_OS_Vertical',
                     width=0.6, range=p.y_range),  source=source, alpha=0.3)
  p.scatter(x=1.5, y=jitter('SP_v_SP_outlier_filtered_OD_Horizontal',
                    width=0.6, range=p.y_range),  source=source, alpha=0.3)
  p.scatter(x=3.5, y=jitter('SP_v_SP_outlier_filtered_OD_Vertical',
                     width=0.6, range=p.y_range),  source=source, alpha=0.3)

  p.x_range.range_padding = 0
  p.xgrid.grid_line_color = None

  return p

#------------------------------------------------------------------------------
# CDS TEST + LR + HV options
#-------------------------------------
# unifying Theme
curdoc().theme = "caliber" # Theme(filename="/content/Theme.yml")
# Theme styling
min_border_right = 20
t_ceil = 20

# default is a vt plot
def get_yt_LR_plot_CDS(e = None, direction = None, eye_v_df = None, eye_x_df = None,
                   vt_OR_xt = "vt", SP_idx_input = None,
                   plot_width=1000, plot_height=400, legend = True, 
                   projection_disabled = False): # e = 0 -> OS / e = 1 -> OD

    # prepare column data source for bokeh engine
    # ColumnDataSource
    eye = output_eye[e] if e is not None else None
    column_name = 'velocity' if (vt_OR_xt == 'vt') else 'position'
    source_df = eye_v_df if (vt_OR_xt == 'vt') else eye_x_df
    source = ColumnDataSource(source_df)
    t = source_df.dropna()['Time']['Time']['Time']
    t_ceil = math.ceil(max(t))  

    # SP_idx 
    SP_index_LH = SP_idx_dict['Left']['Horizontal'][0] #list(SP_idx_input['SP_idx']['OS']['Horizontal'])
    SP_index_LV = SP_idx_dict['Left']['Vertical'][0]#list(SP_idx_input['SP_idx']['OS']['Vertical'])
    SP_index_RH = SP_idx_dict['Right']['Horizontal'][0]#list(SP_idx_input['SP_idx']['OD']['Horizontal'])
    SP_index_RV = SP_idx_dict['Left']['Vertical'][0]#list(SP_idx_input['SP_idx']['OD']['Vertical'])

    # create a view using an filter
    view_LH = CDSView(source=source, filters=[IndexFilter(SP_index_LH)])    
    view_LV = CDSView(source=source, filters=[IndexFilter(SP_index_LV)])  
    view_RH = CDSView(source=source, filters=[IndexFilter(SP_index_RH)])  
    view_RV = CDSView(source=source, filters=[IndexFilter(SP_index_RV)])

    # define figure custom
    # v range as initial
    y_range = (-20,20)
    TOOL_TIPS1 = [( 'time', '@x s'), ('eye velocity', '@y°/s')]
    TOOL_TIPS2 = [( 'horizontal velocity', '@x °/s'), ('vertical velocity', '@y°/s')]
    y_axis_label = "Velocity (°/s)"
    title_projection = 'Eye Velocity Distribution'
    ## default is a vt plot
    if (vt_OR_xt == 'vt'):
      y_range = (-20,20)
      TOOL_TIPS1 = [( 'time', '@x s'), ('eye velocity', '@y°/s')]
      TOOL_TIPS2 = [( 'horizontal velocity', '@x °/s'), ('vertical velocity', '@y°/s')]
      y_axis_label = "Velocity (°/s)"
      title_projection = 'Eye Velocity Distribution'
    # x range
    if (vt_OR_xt == 'xt'):
      y_range = (-30,30) 
      TOOL_TIPS1 = [( 'time', '@x s'), ('eye position', '@y°')]
      TOOL_TIPS2 = [( 'horizontal position', '@x °'), ('vertical position', '@y°')]
      y_axis_label = "Eye Position (°)"
      title_projection = 'Eye Position Track'
    # x axis for projection
    x_axis_label = {False: "Time (s)", True: y_axis_label}[projection_disabled is False]
    title = eye
    left_eye_alpha = .4
    right_eye_alpha = .4

    # geom/glyph scale
    base= plot_width*plot_height/400/300*4/3
    circle_line_ratio = 18
    line_width = 1.8/base #base*len(x)/plot_width
    radius = line_width/circle_line_ratio #len(x)/(plot_width*base*6)

    plot_size_and_tools = {'height': plot_height, 'width': plot_width,
                        'tools':['box_select', 'reset', 'help'],
                        'x_axis_label': x_axis_label, 'x_range': (0, t_ceil),
                        'y_axis_label': y_axis_label, 'y_range': y_range,
                        'tooltips': TOOL_TIPS1 if (projection_disabled is True) else TOOL_TIPS2,
                        'min_border_right':min_border_right,
                        'toolbar_location': None,
                        'output_backend':'svg',
                        'match_aspect': True} 
    auxiliary_line_style = {'line_color':'gray', 'alpha':0.25}
    p0 = figure(title = eye, **plot_size_and_tools, sizing_mode = 'fixed') 
    p_pj = figure(title = title_projection, **plot_size_and_tools, sizing_mode = 'fixed')
  # add multiple renderers
  #---------------------------------------------------
    # add baseline
    p0.line(x=[0,max(t)], 
            y = 0, line_dash="dashed", line_width=line_width, line_color = "black", 
            alpha =0.25)
  #------------------------------------------
    def add_glygh_to_p(column_name):
      y = column_name + '_' + eye + '_' + direction if (eye is not None and direction is not None) else ''
      t = source_df.dropna()['Time']['Time']['Time']
      # unifying VNG level 0 line style
      line_width_and_alpha = {'line_width': line_width, 'alpha': .5}

      if eye is None:
      # since eye is None, specify each glyph
        # OS
          # add Horizontal line plot
        if direction is None:
          p0_OS = p0      
          # create y smooth track
          spl_h = CubicSpline(t, source_df.dropna()[column_name]['OS']['Horizontal']) 
          y_horizontal_smooth = spl_h(t)
          p0_OS.line(t, y_horizontal_smooth, legend_label= "OS_Horizontal",
                line_color=line_color_palatte['greens'][0], **line_width_and_alpha)
          p0_OS.circle(x='Time_Time_Time', y = column_name+'_OS_Horizontal', 
                       legend_label="OS_Horizontal",line_color=None, 
                source = source, view = view_LH,
                radius=radius, fill_color = line_color_palatte['greens'][1], 
                alpha=left_eye_alpha)
          # add Vertical line plot      
          # create y smooth track
          spl_v = CubicSpline(t, source_df.dropna()[column_name]['OS']['Vertical']) 
          y_vertical_smooth = spl_v(t) 
          p0_OS.line(t, y_vertical_smooth, legend_label= "OS_Vertical", 
                line_color=line_color_palatte['oranges'][0], **line_width_and_alpha) 
          p0_OS.circle(x='Time_Time_Time', y = column_name+'_OS_Vertical', 
                       line_color=None, legend_label= "OS_Vertical", 
                source = source, view = view_LV,                
                radius=radius, fill_color = line_color_palatte['oranges'][1], 
                alpha=left_eye_alpha)
          # OD
          # add Horizontal line plot
          p0_OD = p0
          # create y smooth track
          spl_h = CubicSpline(t, source_df.dropna()[column_name]['OD']['Horizontal']) 
          y_horizontal_smooth = spl_h(t)
          p0_OD.line(t, y_horizontal_smooth, legend_label= "OD_Horizontal",
                line_color=line_color_palatte['greens'][3], **line_width_and_alpha)
          p0_OD.circle(x='Time_Time_Time', y = column_name+'_OD_Horizontal', 
                       legend_label="OD_Horizontal",line_color=None, 
                source = source, view = view_RH,
                radius=radius, fill_color = line_color_palatte['greens'][2], 
                alpha=right_eye_alpha)
          # add Vertical line plot
          # create y smooth track      
          spl_v = CubicSpline(t, source_df.dropna()[column_name]['OD']['Vertical']) 
          y_vertical_smooth = spl_v(t) 
          p0_OD.line(t, y_vertical_smooth, legend_label= "OD_Vertical", 
                line_color=line_color_palatte['oranges'][3], **line_width_and_alpha) 
          p0_OD.circle(x='Time_Time_Time', y = column_name+'_OD_Vertical', 
                       line_color=None, legend_label= "OD_Vertical", 
                source = source, view = view_RV,                  
                radius=radius, fill_color = line_color_palatte['oranges'][3], 
                alpha=right_eye_alpha)
          return p0_OS , p0_OD
        else:
          p0_H = p0
          # add Horizontal line plot
          # create y smooth track
          spl = CubicSpline(t, source_df.dropna()[column_name][eye]['Horizontal']) 
          y_horizontal_smooth = spl(t)
          p0_H.line(t, y_horizontal_smooth, legend_label= direction,
                line_color= 'black', **line_width_and_alpha)
          p0_H.circle(x='Time_Time_Time', y = column_name+'_'+eye+'Horizontal', 
                      legend_label=direction,line_color=None, 
                source = source,
                radius=radius, fill_color = 'black', 
                alpha=0.9)
          p0_V = p0
          # add Vertical line plot
          # create y smooth track
          spl = CubicSpline(t, source_df.dropna()[column_name][eye]['Vertical']) 
          y_horizontal_smooth = spl(t)
          p0_V.line(t, y_horizontal_smooth, legend_label= direction,
                line_color= 'black', **line_width_and_alpha)
          p0_V.circle(x='Time_Time_Time', y = column_name+'_'+eye+'Vertical', 
                      legend_label=direction,line_color=None, 
                source = source,
                radius=radius, fill_color = 'black', ##{'field': 'y', 'transform': horizontal_color_mapper}
                alpha=0.9)

          return p0_H , p0_V
      else: # if eye is not none
        if direction is None:
            # add Horizontal line plot 
          p0_H = p0
          # create y smooth track
          spl_h = CubicSpline(t, source_df.dropna()[column_name][eye]['Horizontal']) 
          y_horizontal_smooth = spl_h(t)
          p0_H.line(t, y_horizontal_smooth, legend_label= "Horizontal",
                line_color=line_color_palatte[1], **line_width_and_alpha)
          p0_H.circle(x='Time_Time_Time', y = column_name+'_'+eye+'_Horizontal', 
                      legend_label="Horizontal",line_color=None, 
                source = source,
                radius=radius, fill_color = line_color_palatte[1], 
                alpha=0.9)
            # add Vertical line plot
          p0_V = p0    
          # create y smooth track  
          spl_v = CubicSpline(t, source_df.dropna()[column_name][eye]['Vertical']) 
          y_vertical_smooth = spl_v(t) 
          p0_V.line(t, y_vertical_smooth, legend_label= "Vertical", 
                line_color=line_color_palatte[2], **line_width_and_alpha) 
          p0_V.circle(x='Time_Time_Time', y = column_name+'_'+eye+'_Vertical', 
                      line_color=None, legend_label= "Vertical", 
                source = source,                   
                radius=radius, fill_color = line_color_palatte[3], 
                alpha=0.9)
          return p0_H , p0_V
        else: # if direction is not none
          # create y smooth track 
          spl = CubicSpline(t, source_df.dropna()[column_name][eye][direction]) 
          y_horizontal_smooth = spl(t)
          p0.line(t, y_horizontal_smooth, legend_label= direction,
                line_color= 'black', **line_width_and_alpha)
          p0.circle(x='Time_Time_Time', y = y, legend_label=direction,line_color=None, 
                source = source,
                radius=radius, fill_color = 'black', ##{'field': 'y', 'transform': horizontal_color_mapper}
                alpha=0.9) 
          return p0             
    if (projection_disabled is False): # Do projection on t axis
      print("adjusting style of projection")
      # projection plot setting
      p_pj.plot_width = int(plot_height*1.05) # for viewing a '1:1 feel' aspect ratio
      p_pj.plot_height = plot_height
      p_pj.x_range.start = -20 if (vt_OR_xt == 'vt') else -30
      p_pj.x_range.end = 20 if (vt_OR_xt == 'vt') else 30
      p_pj.y_range.start = -20 if (vt_OR_xt == 'vt') else -30
      p_pj.y_range.end = 20 if (vt_OR_xt == 'vt') else 30
      p_pj.aspect_scale = 1
      # set p scale to 1px/1px
      # aux line
      if (vt_OR_xt == 'vt'):
        p_pj.circle(0, 0, radius = 3.01, fill_color = None, **auxiliary_line_style)
        p_pj.circle(0, 0, radius = 10, fill_color = None, **auxiliary_line_style)
        p_pj.circle(x='velocity_OS_Horizontal', y = 'velocity_OS_Vertical',
              source = source,   
              line_color=None, legend_label= 'OS', radius=radius*7, fill_color = 'blue', 
              alpha =0.2)
        p_pj.circle(x='velocity_OD_Horizontal', y = 'velocity_OD_Vertical',
              source = source,   
              line_color=None, legend_label= 'OD', radius=radius*7, fill_color = 'red', 
              alpha =0.2)
      if (vt_OR_xt == 'xt'):
        p_pj.circle(x='position_OS_Horizontal', y = 'position_OS_Vertical',
              source = source, 
              line_color=None, radius=radius*7, legend_label= 'OS',
              fill_color = 'blue', alpha =0.2)
        p_pj.circle(x='position_OD_Horizontal', y = 'position_OD_Vertical',
              source = source, 
              line_color=None, radius=radius*7, legend_label= 'OD',
              fill_color = 'red', alpha =0.2)        
      p_pj.sizing_mode="fixed"
      p_pj.xaxis.minor_tick_line_color = None
      p_pj.yaxis.minor_tick_line_color = None
      p_pj.output_backend = "svg"
      #p_pj.xgrid.grid_line_color = None
      #p_pj.ygrid.grid_line_color = None
    else: # Do simple yt plot
      print("adjusting style of normal plot")
    # if not needing additional legends
      p0.axis.axis_label_text_font_style = "normal"
      p0.yaxis.minor_tick_line_color = None
      p0.min_border_right = min_border_right
      # styling
      p0.toolbar_location = None

      # Legend border styling
      p0.legend.border_line_color = None
      p0.legend.location = 'top_left'
      p0.legend.border_line_alpha = 1
      p0.output_backend = "svg"

    if (legend == False):
      p0.legend.visible = False
      return p0, p_pj

    add_glygh_to_p(column_name)
    # define output type
    return p0, p_pj

#------------------------------------------------------------------------------
def add_abline_and_annotation_to_p_by_test(p0, p_pj, test_num = None):
  if (test_num ==2):
    # x-t plot 
    gaze_type = ['Up','Down','Right','Left']
    p0.line(x= [0, max(source_df.dropna()['Time']['Time']['Time'])], 
            y = 5, line_dash="dotted", legend_label= 'Gaze Up', 
            line_width=line_width, line_color = "black", alpha =0.25)
    p0.line(x= [0, max(source_df.dropna()['Time']['Time']['Time'])], 
            y = -5, line_dash="dotted", legend_label= 'Gaze Down',
            line_width=line_width, line_color = "black", alpha =0.25)  
    p0.line(x= [0, max(source_df.dropna()['Time']['Time']['Time'])], 
            y = -7.5, line_dash="dashed", legend_label= 'Gaze Right', 
            line_width=line_width, line_color = "black", alpha =0.25)
    p0.line(x=-[0, max(source_df.dropna()['Time']['Time']['Time'])], 
            y = 7.5, line_dash="dashed", legend_label= 'Gaze Left', 
            line_width=line_width, line_color = "black", alpha =0.25)
    # projection x-x
    p_pj.line(x=y_range, y = 5, line_dash="dotted", legend_label= 'Gaze Boundary', 
            line_width=line_width, line_color = "black", alpha =0.25)
    p_pj.line(x=y_range, y = -5, line_dash="dotted", legend_label= 'Gaze Boundary',
            line_width=line_width, line_color = "black", alpha =0.25)
    p_pj.line(x= 7.5, y = y_range, line_dash="dashed", legend_label= 'Gaze Boundary', 
            line_width=line_width, line_color = "black", alpha =0.25)
    p_pj.line(x=-7.5, y = y_range, line_dash="dashed", legend_label= 'Gaze Boundary', 
            line_width=line_width, line_color = "black", alpha =0.25)

#------------------------------------------------------------------------------
def get_raw_xt_and_vt_in_order(vt_OR_xt, test_str = None):
  p_sum = [get_yt_LR_plot_CDS(eye_v_df = eye_v_df, eye_x_df = eye_x_df, 
                              vt_OR_xt = k, SP_idx_input = sp_idx_df,
                              legend = True) for k in ('xt','vt')]
  p1, p2 = p_sum
  p0, p_pj = p1 if (vt_OR_xt=='xt') else p2

  if (test_str is not None):
    test_num = which_test_seq(test_str)
    add_abline_and_annotation_to_p_by_test(p0, p_pj, test_num)
  return p0, p_pj

#------------------------------------------------------------------------------
# Function used to change Text and layout according to test:
 
def choose_test_layout(test_str = None, legend = True, 
                       preset_time_ceil = t_ceil, width =1550):
  test_num = which_test_seq(test_str)
  print(test_num)

  # make subplot
  # PROVIDE preset test time
  test_1_preset_time = 20
  test_2_preset_time = 90
  test_3_preset_time = 60

  row1 = Div(text="")
  row2 = Div(text="")
  row3 = Div(text="")
  row4 = Div(text="")
  row5 = Div(text="")
  row6 = Div(text="")
  row7 = Div(text="")
  row8 = Div(text="")
  lorem_block = Div(text = "")

  # define layout information according to test
  if (test_num == [0]):
      print("Test number not defined")

  # Test 1 header
  elif (test_num == 1):
      # Subplot layout
      print("working on Test 1 plot")
      
      preset_time_ceil = test_1_preset_time
      row1_num = Div(text="<h1>Test 1<h1/>")
      row1_title = Div(text="<h1>Spontaneous Nystagmus<h1/>")
      row1 = row(row1_num, row1_title, width = width) # define initial pixel width
      row2 = Div(text="<h3>No Fixation<h3/>" + 
                 "<p>The patient will be asked to look in a forward direction without fixation of the eyes (i.e. in rest).<p/>")
  # Test 2 header
  elif (test_num == 2):
      # Subplot layout
      print("working on Test 2 plot")
      # 2)    Assessment of the eye (tracking) movements and nystagmus in the (extreme) corners 
      # Duration: circa 60-90 seconds.
      preset_time_ceil = test_1_preset_time   
      
      # Description of the test: The patient will be asked to follow the physician’s finger. 
      # With the start of the test the finger will be placed in the center of the patient’s field of view.
      #// 這裡需要有程式幫忙偵測 Gaze Fixation, Gaze Horizontal, Gaze Vertical, 與 Pursuit的時間分段        
      # From there the physician will slowly and smoothly move his/her finger in the following directions: 
      # 1. (horizontally) to the right (patient perspective), 
      # 2. back to the center, 
      # 3. (horizontally) to the left (patient perspective), 
      # 4. back to the center, 
      # 5. upwards, 
      # 6. back to the center, 
      # 7. downwards, 
      # 8. and for the last time back to the center. 

      # When the patient is asked to follow the physician’s finger both horizontally and vertically, 
      # the movement of the finger will be paused for circa 10 seconds just before the extreme corners  are reached, 
      # in order to assess the nystagmus in this position as well. 

      row1_num = Div(text="<h1>Test 2<h1/>")
      row1_title = Div(text="<h1>Gaze and Pursuit<h1/>")
      row1 = row(row1_num, row1_title, width = width) # define initial pixel width
      row2 = Div(text="<h3>Gaze<h3/>" + 
                 "The patient will be asked to follow the physician’s finger to the extreme corners are reached.")

  # Test 3 header
  elif (test_num == 3):
      preset_time_ceil = test_3_preset_time
      print("working on Test 3 plot")
      # 3)    Test of skew (cover/uncover)

      # Duration: circa 60 seconds. 
      preset_time_ceil = test_3_preset_time
      # Description of the test: The patient will be asked to look in a forward direction. 
      # The physician will cover the patient’s eyes alternately. 
      # The cover up of an eye will take a few seconds every time. 
      # In total the eyes will be covered alternately 3 to 4 times. 
              
      row1_num = Div(text="<h1>Test 3<h1/>")
      row1_title = Div(text="<h1>Test of Skew<h1/>")
      row1 = row(row1_num, row1_title, width = width) # define initial pixel width
      row2 = Div(text="<h3>Gaze + Cover/Uncover <h3/>" + 
                 "The patient will be asked to look in a forward direction. The physician will cover the patient’s eyes alternately.")

      ## Test 3 doesn't need velocity
 
  # Test 4 header
  elif (test_num == 4):
      print("working on Test 4 plot")
      # 4)    Video Head Impulse Test (vHIT)
      # Duration: circa 90-120 seconds. 
      preset_time_ceil = test_4_preset_time        
      # Description of the test: The physician will be asked to perform the HIT twice towards both sides. First two times to the left, than two times to the right.  

      # This part is place holder, TBD
  # add plots
  row3 = Div(text = '<h3>Eye Position<h3/>')
  row4 = row(get_raw_xt_and_vt_in_order('xt')[0], get_raw_xt_and_vt_in_order('xt')[1], lorem_block)
  row5 = Div(text = '<h3>Eye Velocity<h3/>')  
  row6 = row(get_raw_xt_and_vt_in_order('vt')[0], get_raw_xt_and_vt_in_order('vt')[1], get_spv_scatter_stat_by_group(eye_v_outrm_abs_df))
  #row7 = Div(text = '<h3>SPV Analysis<h3/>')

  out = column(row1, row2, row3, row3, row4, row5, row6, row7, row8)    
  #show(out)
  print('Getting'+pkl_path+"'s SVG file")

  # save svg
  # if need_svg == True:
  #   export_svg(out, webdriver = wd, filename = pkl_path[:-18]+'pages.svg')
  #   print('Reading drawing from svg...')
  #   return svg2rlg(pkl_path[:-18]+'pages.svg')
  # else:
  #   print('wait until render complete')
  #   svg = get_svg(out, driver = wd)
  #   print('Reading drawing from svg...')
  #   return svg2rlg(svg)

  export_svg(out, webdriver = wd, filename = pkl_path[:-18]+'pages.svg')
  print('Reading drawing from svg...')
  return svg2rlg(pkl_path[:-18]+'pages.svg')

  return out

#------------------------------------------------------------------------------
def visualization(pkl_path):
    ## Dump variable from pkl_path
    with open(pkl_path,'rb') as f:  # Python 3: open(..., 'rb')
        SPV_mean_dict, SPV_std_dict, SPV_med_dict, SPV_iqr_dict, SPVd_ratio_dict, saccade_num_dict, saccade_num_FR_dict, T, data_m_dict, SP_v_dict, SP_v_SP_outlier_filtered_dict, SP_idx_dict = pickle.load(f)

    # prepare time series
    fps = 210.3
    T= T/fps

    # Create real-world data frame
    # velocity
    eye_v_df = nested_dict_to_pd_df(SP_v_dict)[0]
    # position
    eye_x_df = nested_dict_to_pd_df(data_m_dict)[0]
    # outlier_rm
    eye_v_outrm_df = nested_dict_to_pd_df(SP_v_SP_outlier_filtered_dict)[0]
    # sp_idx
    sp_idx_df = nested_dict_to_pd_df(SP_idx_dict, as_idx=True)[0]
    print(eye_v_df)

    # For statitics compute 
    eye_v_outrm_abs_df = eye_v_outrm_df.abs()
    print(eye_v_outrm_abs_df)


    output_notebook()
    drawing = choose_test_layout(pkl_path)
    print('drawing is '+ str(drawing))
    print('Scaling SVG image...')
    scale_x = scale_y = 0.35  # scaling factor
    drawing.width = drawing.minWidth() * scale_x
    drawing.height = drawing.height * scale_y
    drawing.scale(scale_x, scale_y)

    data = [
        ['NeuroSpeed System - Test Result',None],  # this works as a leading row
        [drawing,None]
    ]
    table = Table(# Flowable object
        data,
        #colWidths=400,
        rowHeights=[1 * mm] + [drawing.height] * (len(data) - 1),
        hAlign='LEFT',
        repeatRows=1
    )
    table.setStyle(TableStyle([
        # LEADING ROW
        ('LINEBELOW', (0, 0), (1, 0), 0.5, colors.black),
        ('SPAN', (0, 0), (1, 0)),                           # colspan
        ('FONTSIZE', (0, 0), (1, 0), 12),
        ('LEFTPADDING', (0, 0), (1, 0), 0),
        ('BOTTOMPADDING', (0, 0), (1, 0), 3),

        # REST OF ROWS
        ('LEFTPADDING', (1, 1), (-1, -1), 0),
        ('RIGHTPADDING', (1, 1), (-1, -1), 0),
        ('BOTTOMPADDING', (1, 1), (-1, -1), 0),
        ('TOPPADDING', (1, 1), (-1, -1), 0),
    ]))
    margin = 15 * mm

    # path 
    doc = BaseDocTemplate(pkl_path[:-18]+'pages.pdf', pagesize=A4, rightMargin=margin, leftMargin=margin, topMargin=margin, bottomMargin=margin)


    #portrait_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='portrait_frame ')
    landscape_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.height, doc.width, id='landscape_frame ') #SimpleCocTemplate

    print('Making story...')
    story = []
    story.append(NextPageTemplate('landscape'))
    story.append(table)
    #story.append(PageBreak())
    #story.append(NextPageTemplate('landscape'))

    # story.append(<next page content>)
    # story.append(PageBreak())
    # doc.addPageTemplates([PageTemplate(id='portrait',frames=portrait_frame),
    #                       PageTemplate(id='landscape',frames=landscape_frame, pagesize=landscape(A4)),
    #                       ])

    doc.addPageTemplates([PageTemplate(id='landscape',frames=landscape_frame, pagesize=landscape(A4))])
    print('Building document...')
    doc.build(story)