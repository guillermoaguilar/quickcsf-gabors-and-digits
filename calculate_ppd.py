#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculates PPD for setup in TM lab.


@author: G. Aguilar, April 2026
"""

import math

screen_width_px   = 1920  # horizontal resolution
screen_width_cm   = 52    # physical screen width in cm
viewing_dist_cm   = 80    # viewing distance in cm

px_per_cm = screen_width_px / screen_width_cm
ppd = px_per_cm * viewing_dist_cm * math.tan(math.radians(1))
print(f"PPD = {ppd:.2f}")

# 