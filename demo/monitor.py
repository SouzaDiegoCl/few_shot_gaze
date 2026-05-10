#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import os
import re
import subprocess
import numpy as np

try:
    import gi
    gi.require_version('Gdk', '3.0')
    from gi.repository import Gdk
except ImportError:
    Gdk = None

class monitor:

    def __init__(self):
        if Gdk is not None:
            display = Gdk.Display.get_default()
            if display is not None:
                screen = display.get_default_screen()
                default_screen = screen.get_default()
                num = default_screen.get_number()

                self.h_mm = default_screen.get_monitor_height_mm(num)
                self.w_mm = default_screen.get_monitor_width_mm(num)

                self.h_pixels = default_screen.get_height()
                self.w_pixels = default_screen.get_width()
                return

        self._set_fallback_geometry()

    def _set_fallback_geometry(self):
        self.w_pixels = 1920
        self.h_pixels = 1080
        self.w_mm = 344.0
        self.h_mm = 194.0

        try:
            output = subprocess.check_output(
                "xrandr 2>/dev/null | awk '/ connected/{print; exit}'",
                shell=True,
            ).decode("utf-8", "ignore")

            pixel_match = re.search(r"(\d+)x(\d+)\+\d+\+\d+", output)
            if pixel_match is None:
                pixel_match = re.search(r"(\d+)x(\d+)", output)
            if pixel_match is not None:
                self.w_pixels = int(pixel_match.group(1))
                self.h_pixels = int(pixel_match.group(2))

            mm_match = re.search(r"(\d+)mm x (\d+)mm", output)
            if mm_match is not None:
                self.w_mm = float(mm_match.group(1))
                self.h_mm = float(mm_match.group(2))
        except Exception:
            pass

    def monitor_to_camera(self, x_pixel, y_pixel):

        # assumes in-build laptop camera, located centered and 10 mm above display
        # update this function for you camera and monitor using: https://github.com/computer-vision/takahashi2012cvpr
        x_cam_mm = ((int(self.w_pixels/2) - x_pixel)/self.w_pixels) * self.w_mm
        y_cam_mm = 10.0 + (y_pixel/self.h_pixels) * self.h_mm
        z_cam_mm = 0.0

        return x_cam_mm, y_cam_mm, z_cam_mm

    def camera_to_monitor(self, x_cam_mm, y_cam_mm):
        # assumes in-build laptop camera, located centered and 10 mm above display
        # update this function for you camera and monitor using: https://github.com/computer-vision/takahashi2012cvpr
        x_mon_pixel = np.ceil(int(self.w_pixels/2) - x_cam_mm * self.w_pixels / self.w_mm)
        y_mon_pixel = np.ceil((y_cam_mm - 10.0) * self.h_pixels / self.h_mm)

        x_mon_pixel = int(np.clip(x_mon_pixel, 0, self.w_pixels - 1))
        y_mon_pixel = int(np.clip(y_mon_pixel, 0, self.h_pixels - 1))

        return x_mon_pixel, y_mon_pixel
