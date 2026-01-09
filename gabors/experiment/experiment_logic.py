import sys
import time

import numpy as np
import stimuli
import text_displays


def display_fixation_cross(ihrl, intensity=0):
    ihrl.graphics.flip(clr=True)
    fix = ihrl.graphics.newTexture(np.ones((5, 5)) * intensity)
    fix.draw((ihrl.width // 2, ihrl.height // 2))
    ihrl.graphics.flip()
    return


def run_trial(
    ihrl,
    sf,
    orientation,
    phase,
    contrast,
    **kwargs
):
    """Function that runs sequence of events during one trial"""
    window_center = (ihrl.height // 2, ihrl.width // 2)  # Center of the drawing window

    # Fixation cross
    display_fixation_cross(ihrl)

    # Create textures before showing
    gabor1 = stimuli.gabor(sf,
                           contrast=contrast,
                           sigma=0.5, 
                           orientation=orientation, # in degrees
                           phase=phase, # in degrees
                           )                          

    # Convert the stimulus image(matrix) to an OpenGL texture
    gabor_texture = ihrl.graphics.newTexture(gabor1["img"])
    
    # Determine position: we want the stimulus in the center of the frame
    pos = (window_center[1] - (gabor_texture.wdth // 2),
           window_center[0] - (gabor_texture.hght // 2))
    
    time.sleep(0.5)
    
    ### 1st interval
    # draw and flip first interval
    gabor_texture.draw(pos=pos, sz=(gabor_texture.wdth, gabor_texture.hght))
    ihrl.graphics.flip(clr=True)  # flips the frame buffer to show everything
    #ihrl.sounds[0].play(loops=0, maxtime=int(0.5*1000)) # stim time in ms
    time.sleep(0.5) 
        
    ### wait for response
    display_fixation_cross(ihrl, intensity=0)
    btn, t1 = ihrl.inputs.readButton(btns=["Up", "Down", "Escape", "Space"])
    
    if btn=="Up":
        response = 1
        
    elif btn=="Down":
        response = 0

    # Raise SystemExit Exception
    if (btn == "Escape") or (btn == "Space"):
        sys.exit("Participant terminated experiment.")

    # end trial
    return {"response-btn": btn, "response": response, "resp.time": t1}


def display_instructions(ihrl):
    """Display instructions to the participant

    Parameters
    ----------
    ihrl : hrl-object
        hrl-interface object to use for display
    """
    lines = [
        "Contrast detection task",
        "Answer if you saw the stimulus or not",
        "Press UP for YES, I saw it",
        "or",
        "Press DOWN for NO, I did not see it",
        "",
        "Press MIDDLE button to start",
    ]

    text_displays.display_text(
        ihrl=ihrl, text=lines, intensity_background=ihrl.graphics.background
    )

    return
