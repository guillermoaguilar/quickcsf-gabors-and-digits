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
    
    # input is RMS contrast. I need to convert to Michelson contrast for this stimulus
    # the code below finds that a linear relationship exist between these two.
    # thus we just need to invert it.
    # Intercept is zero, so f-1 (y) = 1/m *(y)
    contrast = contrast/0.09405856524907606
    
    
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


# %%
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from scipy import stats
    
    stim = gabor(sf=8, contrast=0.15)
    
    rms = stim['img'].std()
    m = stim['img'].mean()

    print(f"mean: {m.round(4)}")
    print(f"RMS contrast: {rms.round(4)}")
        

    print(stim['img'].shape)
    
    stimupy.utils.plot_stim(stim)
    
# %%
    ## relationship between contrast of gabor (Michelson?) and RMS contrast
    # contrast_vector = np.linspace(0.001, 1.0, 10)
    
    # rms=[]
    # for c in contrast_vector:
    #     print(c)
    #     stim = gabor(sf=16, contrast=c)
    #     rms.append(stim['img'].std())
        
    # rms = np.array(rms)
    # plt.figure()
    # plt.plot(contrast_vector, rms, 'o')
    # plt.show()
    
    # # linear relationship
    # slope, intercept, r, p, se  = stats.linregress(contrast_vector, rms)
    
    #slope
    #Out[12]: 0.09405856524907606

    #intercept
    #Out[13]: 0.0
    
    
