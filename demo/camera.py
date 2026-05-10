#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import cv2
import numpy as np
import pickle
import select
import sys

def cam_calibrate(cam_idx, cap, cam_calib):

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    pts = np.zeros((6 * 9, 3), np.float32)
    pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # capture calibration frames
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to read a frame from the camera.")
            continue

        frame_copy = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retc, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if retc:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(frame_copy, (9, 6), corners, retc)
            status_text = "Chessboard found. Press s to save, c to continue, q to quit."
        else:
            status_text = "Show the chessboard to the camera. Press q to quit."

        cv2.putText(frame_copy, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('points', frame_copy)

        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            key = None

        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                terminal_input = sys.stdin.readline().strip().lower()
                if terminal_input:
                    key = ord(terminal_input[0])
        except (OSError, ValueError, AttributeError):
            pass

        if key == ord('s') and retc:
            img_points.append(corners)
            obj_points.append(pts)
            frames.append(frame)
        elif key == ord('c'):
            continue
        elif key in (ord('q'), 27):
            print("Calibrating camera...")
            cv2.destroyAllWindows()
            break

    # compute calibration matrices

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, frames[0].shape[0:2], None, None)

    # check
    error = 0.0
    for i in range(len(frames)):
        proj_imgpoints, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error += (cv2.norm(img_points[i], proj_imgpoints, cv2.NORM_L2) / len(proj_imgpoints))
    print("Camera calibrated successfully, total re-projection error: %f" % (error / len(frames)))

    cam_calib['mtx'] = mtx
    cam_calib['dist'] = dist
    print("Camera parameters:")
    print(cam_calib)

    pickle.dump(cam_calib, open("calib_cam%d.pkl" % (cam_idx), "wb"))
