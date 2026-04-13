#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculates PPD for setup in TM lab.

Follow stimupy calculation formula


@author: G. Aguilar, April 2026
"""

import math
import numpy as np

screen_width_px   = 1920  # horizontal resolution
screen_width_cm   = 52    # physical screen width in cm
viewing_dist_cm   = 80    # viewing distance in cm

px_per_cm = screen_width_px / screen_width_cm
ppd = px_per_cm * viewing_dist_cm * math.tan(math.radians(1))
print(f"PPD = {ppd:.2f}")



def compute_ppd(screen_size, resolution, distance):
    """Compute the pixels per degree in a presentation setup
    i.e., the number of pixels in the central one degree of visual angle

    Parameters
    ----------
    screen_size : (float, float)
        physical size, in whatever units you prefer, of the presentation screen
    resolution : (float, float)
        screen resolution, in pixels,
        in the same direction that screen size was measured in
    distance : float
        physical distance between the observer and the screen, in the same unit as screen_size

    Returns
    -------
    float
        ppd, the number of pixels in one degree of visual angle
    """

    ppmm = resolution / screen_size
    mmpd = 2 * np.tan(np.radians(0.5)) * distance
    return ppmm * mmpd

ppd2 = compute_ppd(screen_width_cm, screen_width_px, viewing_dist_cm)

print(f"PPD-stimupy = {ppd2:.2f}")


image_size_px = 512/2

image_size_deg = image_size_px / ppd # pix * deg/pix = deg

print(f"an image of size {image_size_px} pixels has size {image_size_deg} degrees")