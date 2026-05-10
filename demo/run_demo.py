#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import time
import cv2
import numpy as np
from os import path
from subprocess import call
import pickle
import sys
import torch
import os

import warnings
warnings.filterwarnings("ignore")

from monitor import monitor
from camera import cam_calibrate
from person_calibration import collect_data, fine_tune
from frame_processor import frame_processer

#################################
# Start camera
#################################

cam_idx = int(os.environ.get("CAMERA_INDEX", "1"))
base_dir = path.dirname(path.abspath(__file__))


def require_file(file_path, message):
    if not path.isfile(file_path):
        raise FileNotFoundError(message)

# adjust these for your camera to get the best accuracy
call('v4l2-ctl -d /dev/video%d -c brightness=100' % cam_idx, shell=True)
call('v4l2-ctl -d /dev/video%d -c contrast=50' % cam_idx, shell=True)
call('v4l2-ctl -d /dev/video%d -c sharpness=100' % cam_idx, shell=True)

cam_cap = cv2.VideoCapture(cam_idx)
cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# calibrate camera
cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}
calib_path = path.join(base_dir, "calib_cam%d.pkl" % (cam_idx))
if path.exists(calib_path):
    cam_calib = pickle.load(open(calib_path, "rb"))
else:
    print("Calibrate camera once. Print pattern.png, paste on a clipboard, show to camera and capture non-blurry images in which points are detected well.")
    print("Press s to save frame, c to continue, q to quit")
    cam_calibrate(cam_idx, cam_cap, cam_calib)

#################################
# Load gaze network
#################################
ted_parameters_path = path.join(base_dir, 'demo_weights', 'weights_ted.pth.tar')
maml_parameters_path = path.join(base_dir, 'demo_weights', 'weights_maml')
k = 9

require_file(
    ted_parameters_path,
    "Missing T-ED weights: {}. Download demo_weights.zip and extract it into demo/.".format(ted_parameters_path)
)

# Set device
force_cpu = os.environ.get("FORCE_CPU", "0").lower() in ("1", "true", "yes")
if force_cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("> Initializing model imports...", flush=True)
print("> Device: %s" % device, flush=True)

# Create network
sys.path.append(path.abspath(path.join(base_dir, "..", "src")))
print("> Importing DTED model class...", flush=True)
from models import DTED
print("> Creating DTED network instance...", flush=True)
gaze_network = DTED(
    growth_rate=32,
    z_dim_app=64,
    z_dim_gaze=2,
    z_dim_head=16,
    decoder_input_c=32,
    normalize_3d_codes=True,
    normalize_3d_codes_axis=1,
    backprop_gaze_to_encoder=False,
).to(device)
print("> DTED network created successfully.", flush=True)

#################################

# Load T-ED weights if available
print('> Loading: %s' % ted_parameters_path, flush=True)
print("> Reading T-ED weights file...", flush=True)
ted_weights = torch.load(ted_parameters_path)
print("> T-ED weights loaded. Checking for device mismatch...", flush=True)
if torch.cuda.device_count() == 1:
    if next(iter(ted_weights.keys())).startswith('module.'):
        ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])
print("> T-ED weights processed.", flush=True)

#####################################

# Load MAML MLP weights if available
full_maml_parameters_path = maml_parameters_path +'/%02d.pth.tar' % k
require_file(
    full_maml_parameters_path,
    "Missing MAML weights: {}. Download demo_weights.zip and extract it into demo/.".format(full_maml_parameters_path)
)
print('> Loading: %s' % full_maml_parameters_path, flush=True)
print("> Reading MAML weights file...", flush=True)
maml_weights = torch.load(full_maml_parameters_path)
print("> MAML weights loaded. Updating state dict...", flush=True)
ted_weights.update({  # rename to fit
    'gaze1.weight': maml_weights['layer01.weights'],
    'gaze1.bias':   maml_weights['layer01.bias'],
    'gaze2.weight': maml_weights['layer02.weights'],
    'gaze2.bias':   maml_weights['layer02.bias'],
})
print("> Applying weights to network...", flush=True)
gaze_network.load_state_dict(ted_weights)
print("> Network weights loaded successfully.", flush=True)

#################################
# Personalize gaze network
#################################

# Initialize monitor and frame processor
print("> Preparing monitor and frame processor...")
mon = monitor()
frame_processor = frame_processer(cam_calib, device=device)

# collect person calibration data and fine-
# tune gaze network
print("> Waiting for subject name. Type it and press Enter.", flush=True)
subject = input('Enter subject name: ')
print("> Starting calibration capture for subject: %s" % subject, flush=True)
data = collect_data(cam_cap, mon, calib_points=9, rand_points=4)
print("> Calibration capture finished.", flush=True)
# adjust steps and lr for best results
# To debug calibration, set show=True
gaze_network = fine_tune(subject, data, frame_processor, mon, device, gaze_network, k, steps=1000, lr=1e-5, show=False)

#################################
# Run on live webcam feed and
# show point of regard on screen
#################################
data = frame_processor.process(subject, cam_cap, mon, device, gaze_network, show=True)
