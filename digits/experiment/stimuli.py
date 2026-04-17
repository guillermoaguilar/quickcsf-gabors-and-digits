import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from stimupy.components.gaussians import gaussian

# Defaults
PPD = 51      # pixels per degree
IMSIZE = 512  # pixels

VIS_SIZE = IMSIZE/PPD  #pixels * degree/pix = degree

########
#SF_CONDITIONS = {
#    0.5: "0.50cpd",
#    1.0: "1.00cpd",
#    2.0: "2.00cpd",
#    4.0: "4.00cpd",
#    8.0: "8.00cpd",
#    15.80: "15.80cpd",
#    }
#
# FOLDER = "filtered"   

######## first design - copy range from gabors
#frequency_vector = np.logspace(np.log10(.2),
#                               np.log10(int(PPD/2)),
#                               25, endpoint=True).round(2)
#FOLDER = "filtered-more-sf"      

######## second design - lower bound adjusted, minimum that is distinguishable  
frequency_vector = np.logspace(np.log10(1.0),
                               np.log10(int(PPD/2)),
                               25, endpoint=True).round(2)
FOLDER = "filtered"  

#######################################################################
# (sf_cpd, label)                        
SF_CONDITIONS = {f: f"{f:.2f}cpd" for f in frequency_vector}

print(SF_CONDITIONS)  

# %% Standard Gabor
def load_digit(
    sf,
    contrast,
    digit, 
    debug=False,
):
    
    label = SF_CONDITIONS[sf]

    # load from file
    im = Image.open(f"stimuli/{FOLDER}/{digit}/{digit}_{label}.png")
    im = np.array(im)/255
    
    # applies Gaussian envelope to avoid edge artifacts at low s.f.
    gaussian_window = gaussian(visual_size=VIS_SIZE,
    				           ppd=PPD,
                               sigma=2.5,
                               origin="center",
                              )["img"][::2, ::2]  # half the size
    
    #im = (im - im.mean()) * gaussian_window + im.mean()
    
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
    #stim = load_digit(sf=1, contrast=0.05, digit=8, debug=True)
    stim = load_digit(sf=0.82, contrast=0.15, digit=2, debug=True)

    plt.imshow(stim, cmap='gray', vmin=0, vmax=1)
    plt.show()
