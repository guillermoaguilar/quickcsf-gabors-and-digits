#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pad images of digits to double their width and height


@author: G. Aguilar, April 2026
"""


from PIL import Image


color = (255, 255, 255)

images = [0, 2, 3, 4, 5, 6, 7, 8, 9]

for num in images:

    im = Image.open(f"original/not-padded/{num}.png")
    
    width, height = im.size
    
    # %%
    new_width = 2*width
    new_height = 2*width
    
    left = int(width/2)
    top = int(width/2)+1
    
    
    result = Image.new(im.mode, (new_width, new_height), color)
    result.paste(im, (left, top))
    result = result.resize((512, 512))
    result.save(f"original/{num}.png")
