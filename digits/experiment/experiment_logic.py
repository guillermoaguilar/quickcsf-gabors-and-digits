import sys
import time

import numpy as np
import stimuli
import text_displays


def display_fixation_cross(ihrl, intensity=0, size=5):
    ihrl.graphics.flip(clr=True)
    fix = ihrl.graphics.newTexture(np.ones((size, size)) * intensity)
    fix.draw((ihrl.width // 2, ihrl.height // 2))
    ihrl.graphics.flip()
    return


def run_trial(
    ihrl,
    sf,
    contrast,
    digit,
    **kwargs
):
    """Function that runs sequence of events during one trial"""
    window_center = (ihrl.height // 2, ihrl.width // 2)  # Center of the drawing window

    # Fixation cross - blanca
    display_fixation_cross(ihrl, intensity=1.0, size=10)

    # Create textures before showing
    stim = stimuli.load_digit(sf=sf,
                              contrast=contrast,
                              digit=digit, # which digit to show, as int
                              )                          

    # Convert the stimulus image(matrix) to an OpenGL texture
    stim_texture = ihrl.graphics.newTexture(stim)
    
    # Determine position: we want the stimulus in the center of the frame
    pos = (window_center[1] - (stim_texture.wdth // 2),
           window_center[0] - (stim_texture.hght // 2))
    
    ### time of fixation
    time.sleep(0.5)
    
    
    # draw and flip
    stim_texture.draw(pos=pos, sz=(stim_texture.wdth, stim_texture.hght))
    ihrl.graphics.flip(clr=True)  # flips the frame buffer to show everything
    #ihrl.sounds[0].play(loops=0, maxtime=int(0.5*1000)) # stim time in ms
    time.sleep(0.3) 
    
    # limpia
    ihrl.graphics.flip(clr=True)
    time.sleep(0.1) 
    
    ## fixation negra, espera respuesta
    display_fixation_cross(ihrl, intensity=0.0, size=10)
    
    ### wait for response
    btn, t1 = ihrl.inputs.readButton(btns=["Up", "Down", "Escape", "Space"])
    
    ## TODO: see if response was indeed correct
    
    ## response --> correct or incorrect, binary
    if btn=="Up":
        response = 1
        
    elif btn=="Down":
        response = 0

    # Raise SystemExit Exception
    if (btn == "Escape") or (btn == "Space"):
        sys.exit("Participant terminated experiment.")

    # after response, sleep for 1 second
    ihrl.graphics.flip(clr=True) 
    time.sleep(0.5)
    


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
        "Medición de sensibilidad al contraste",
        "Responda que digito vio",
        "",
        "Presiona ESPACIO para empezar",
    ]

    text_displays.display_text(
        ihrl=ihrl, text=lines, intensity_background=ihrl.graphics.background
    )

    return
