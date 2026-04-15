from hrl import HRL
import numpy as np

from stimuli import load_digit, SF_CONDITIONS, frequency_vector, PPD
from stimupy.components import texts

# Possible digits
digits = [0, 2, 3, 4, 5, 6, 7, 8, 9]

CONTRAST_FIXED = 0.1                                                        
                              
def stimuli():
    """ Loads all images of digits into a dictionary"""
    stims={}
    
    for digit in digits:
        for sf in frequency_vector:
            stim = load_digit(sf=sf,
                              contrast=CONTRAST_FIXED ,
                              digit=digit, # which digit to show, as int
                              )
            stims[f"{digit}_{sf}cpd"] = stim
                                                    
    return stims  


def display_stim(ihrl, stim_image, stim_name):
    """
    In this "experiment", we just display a collection of stimuli, one at a time.
    Here we define a function to display a single stimulus image centrally on the screen.
    """
    window_center = (ihrl.height // 2, ihrl.width // 2)  # Center of the drawing window

    # Convert the stimulus image(matrix) to an OpenGL texture
    stim_texture = ihrl.graphics.newTexture(stim_image)

    # Determine position: we want the stimulus in the center of the frame
    pos = (window_center[1] - (stim_texture.wdth // 2),
           window_center[0] - (stim_texture.hght // 2))
           
    # Create a display: draw texture on the frame buffer
    stim_texture.draw(pos=pos, sz=(stim_texture.wdth, stim_texture.hght))
    
    # Display name of stimulus
    label = f"{stim_name} - RMS contrast {CONTRAST_FIXED}"
    
    text_arr = texts.text(text=label,
                          intensity_text=0.0,
                          intensity_background=0.5,    
                          ppd=PPD,
                          fontsize=40,
                         )["img"]
    textline = ihrl.graphics.newTexture(text_arr)
    
    text_pos = (50, 50)
    textline.draw(pos=text_pos)
    
    # Display: flip the frame buffer
    ihrl.graphics.flip()  # also `clear` the frame buffer

    return

def select(ihrl, value, rng):
    """Allow participant to select a value from a range of options

    Parameters
    ----------
    ihrl : hrl-object
        HRL-interface object to use for display
    value : int
        currently selected option
    rng : (int, int)
        min and max values to select. If one value is given, assume min=0

    Returns
    -------
    int
        currently selected option
    bool
        whether this option was confirmed

    Raises
    ------
    SystemExit
        if participant/experimenter terminated by pressing Escape
    """
    try:
        len(rng)
    except:
        rng = (0, rng)

    accept = False

    press, _ = ihrl.inputs.readButton(btns=("Left", "Right", "Escape", "Space"))

    if press == "Escape":
        # Raise SystemExit Exception
        sys.exit("Participant terminated experiment.")
    elif press == "Left":
        value -= 1
        value = max(value, rng[0])
    elif press == "Right":
        value += 1
        value = min(value, rng[1])
    elif press == "Space":
        accept = True

    return value, accept
    
 
def experiment_main(ihrl):
    
    stims = stimuli()
    stim_names = [*stims.keys()]
    
    print(f"Stimuli available: {stim_names}")
    stim_idx = 0
    while True:
        # Main loop
        try:
            # Display stimulus
            stim_name = stim_names[stim_idx]
            print(f"Showing {stim_name}")
            
            stim_image = stims[stim_name]
            display_stim(ihrl, stim_image, stim_name)
                        
            # Select next stim
            stim_idx, _ = select(ihrl, value=stim_idx, rng=len(stim_names) - 1)
            
        except SystemExit as e:
            # Cleanup
            print("Exiting...")
            ihrl.close()
            raise e    
    

if __name__ == "__main__":
   
    ihrl = HRL(
        graphics="gpu",  # Use the default GPU as graphics device driver
        inputs="keyboard",  # Use the keyboard as input device driver
        hght=1080,
        wdth=1920,
        scrn=0,  # Which screen (monitor) to use
        fs=True,
        bg=0.5,  # background intensity (black=0.0; white=1.0)
        lut='lut.csv',
    )
    experiment_main(ihrl)
    ihrl.close()   
    
