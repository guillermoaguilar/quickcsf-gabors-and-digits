import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Defaults
PPD = 62      # pixels per degree
IMSIZE = 512 # pixels

VIS_SIZE = IMSIZE/PPD  #pixels * degree/pix = degree

SF_CONDITIONS = {
    0.5: "0.50cpd",
    1.0: "1.00cpd",
    2.0: "2.00cpd",
    4.0: "4.00cpd",
    8.0: "8.00cpd",
    15.80: "15.80cpd",
    }


# %% Standard Gabor
def load_digit(
    sf,
    contrast,
    digit, 
    debug=False,
):
    
    label = SF_CONDITIONS[sf]

    # load from file
    im = Image.open(f"stimuli/filtered/{digit}/{digit}_{label}.png")
    im = np.array(im)/255
    
    # adjust contrast
    rms = im.std()
    m = im.mean()
    if debug:
        print(f"mean original: {m.round(4)}")
        print(f"RMS contrast original: {rms.round(4)}")
    
    k = contrast / rms # ajusting factor
    scaled_im = m + k * (im - m)
    
    if debug:
        print(f"mean adjusted: {scaled_im.mean().round(4)}")
        print(f"RMS contrast afterwards: {scaled_im.std().round(4)}")
    
    return scaled_im



if __name__ == "__main__":
    stim = load_digit(sf=1, contrast=0.05, digit=8, debug=True)

    plt.imshow(stim, cmap='gray', vmin=0, vmax=1)
    plt.show()
