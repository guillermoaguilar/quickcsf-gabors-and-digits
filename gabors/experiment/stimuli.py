import numpy as np
import stimupy

# Defaults
IM_SIZE = 5   # deg. visual angle of the bounding square
PPD = 51      # pixels per degree.

INTENSITY_BACKGROUND = 0.5


# %% Standard Gabor
def gabor(
    sf,
    contrast,
    sigma=0.75, 
    orientation=0, # in degrees
    phase=0, # in degrees
    intensity_background=INTENSITY_BACKGROUND,
):

    visual_size = IM_SIZE
    
    # min and max values of the sinusoid
    l1 = intensity_background - contrast/2
    l2 = intensity_background + contrast/2
    
    return stimupy.stimuli.gabors.gabor(
           visual_size=visual_size, 
           ppd=PPD, 
           frequency=sf,
           rotation=orientation, 
           phase_shift=phase, 
           intensities=(l1, l2), 
           sigma=sigma)



if __name__ == "__main__":
    stim = gabor(sf=4, contrast=1)

    print(stim['img'].shape)
    
    stimupy.utils.plot_stim(stim)
