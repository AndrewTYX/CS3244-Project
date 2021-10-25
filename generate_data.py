import sys
import os
import pandas as pd
from glob import glob
import numpy as np
import scipy as sp
import tensorflow as tf
import pydicom
import os
import cv2

BAD_IDS = {'ID00011637202177653955184', 'ID00052637202186188008618'}
ct_data_path = './train'

def generate_patient_ids(dir_path):
  '''
  Generate patient IDs
  '''
  res_list = []
  folders = glob(os.path.join(dir_path, "*"))
  for folder in folders:
    res_list.append(os.path.basename(folder))
  return res_list


def get_base_int(path):
  name = os.path.basename(path)
  num = name[:-4]
  return int(num)

def read_ct(path):
  dicom = pydicom.read_file(path)
    
  data = dicom.pixel_array
        
  return data

def build_ct_3d_for_patient(ct_data_path, patient):
  '''
  Returns combined np array for CT scan for 1 patient
  
  Output shape: (n , ct_shape[0], ct_shape[1])
  '''
  # try:
  patient_ct = glob(os.path.join(ct_data_path, patient, '*.dcm'))
  patient_ct = sorted(patient_ct, key = get_base_int)
  ct_arr = read_ct(patient_ct[0])
  ct_shape = ct_arr.shape
  arr = np.zeros((len(patient_ct), ct_shape[0], ct_shape[1]))
  i = 0
  for ct_path in patient_ct:
    ct_arr = read_ct(ct_path)
    arr[i] = ct_arr
    i += 1
  return arr

def rs_img(patient_ID):
    '''
    W, H be 512 now
    returns all 2D slices of the patient
    '''
    w, h = 512, 512
    patient_ct = build_ct_3d_for_patient(ct_data_path, patient_ID)
    resized_scans = [cv2.resize(patient_ct[i], (w, h), interpolation = cv2.INTER_AREA) for i in range(len(patient_ct))]
    return np.array(resized_scans)

def even_slice_select(combined_arr, N):
  assert combined_arr.shape[0] >= N
  selected_slices = []
  target_depth = N
  depth = combined_arr.shape[0]
  depth_factor = depth / target_depth

  i = 0
  while (i < target_depth):
    selected_slices.append(combined_arr[round(i * depth_factor)])
    i += 1
  
  selected_slices = np.array(selected_slices)
  return selected_slices

def spline_interpolated_zoom_select(combined_arr, N):
    selected_slices = []
    target_depth = N
    depth = combined_arr.shape[0]
    depth_factor = target_depth / depth

    selected_slices =  sp.ndimage.interpolation.zoom(combined_arr, [depth_factor, 1, 1])
    return selected_slices

def rotate(img3d):
  num_cols = img3d.shape[2]
  rotated = []
  for i in range(num_cols):
    rotated.append(np.transpose(img3d[:, :, i]))
  return np.array(rotated)

def save_np_arr_with_channel(channel_num):
    raw_arr_dir = {}
    patients_list = generate_patient_ids(ct_data_path)
    length = len(patients_list) - 2
    sample_ct_shape = rotate(spline_interpolated_zoom_select(rs_img(patients_list[0]), channel_num)).shape
    pixel_arr_combined = np.empty((length, sample_ct_shape[0], sample_ct_shape[1], sample_ct_shape[2]))
    index = 0
    for patient in patients_list:
      if patient not in BAD_IDS:
        print(f'Processing patient: {patient}')
        pixel_arr_combined[index] = rotate(spline_interpolated_zoom_select(rs_img(patient), channel_num))
        raw_arr_dir[patient] = rotate(spline_interpolated_zoom_select(rs_img(patient), channel_num))
        index = index + 1;
    np.save(f'./ct_interpolated_{channel_num}', pixel_arr_combined)
    np.save(f'./ct_interpolated_{channel_num}_dir', raw_arr_dir)
