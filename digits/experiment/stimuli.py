import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Defaults
PPD = 62      # pixels per degree.

INTENSITY_BACKGROUND = 0.5

SF_CONDITIONS = {
    0.5: "12deg_0.50cpd",
    1.0: "06deg_1.00cpd",
    2.0: "03deg_2.00cpd",
    4.0: "1.5deg_4.00cpd",
    8.0: "0.75deg_8.00cpd",
    15.80: "0.38deg_15.8cpd",
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
        print(f"mean original: {m}")
        print(f"RMS contrast original: {rms}")
    
    k = contrast / rms # ajusting factor
    scaled_im = m + k * (im - m)
    
    if debug:
        print(f"mean adjusted: {scaled_im.mean()}")
        print(f"RMS contrast afterwards: {scaled_im.std()}")
    
    return scaled_im



if __name__ == "__main__":
    stim = load_digit(sf=2, contrast=0.05, digit=8, debug=True)

    plt.imshow(stim, cmap='gray', vmin=0, vmax=1)
    plt.show()
