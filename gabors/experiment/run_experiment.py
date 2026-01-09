#!/usr/bin/env python
"""
Contrast detection in a Yes-No task. The contrast and frequency is 
adaptively ajusted using the quickCSF algorithm (Lesmes et al. 2010)

Uses HRL on python 3

@authors: G. Aguilar, Jan 2026
"""
import random
from math import pi
import pygame
import pandas as pd
from socket import gethostname
from hrl import HRL

#from helper_functions import make_sound
import data_management
import text_displays
import experiment_logic

from qCSF import qCSF

NTRIALS = 10
SUFFIX = 'GABORS'

# Possible orientations
orientations = [0, 90]

# Possible phases
phases = [0, pi/4, pi/2, 3*pi/4]

# Initialize quickCSF object
qcsf = qCSF()


if "vlab" in gethostname():
    SETUP = {
        "graphics": "datapixx",
        "inputs": "responsepixx",
        "scrn": 1,
        "lut": "lut.csv",
        "fs": True,
        "wdth": 1024,
        "hght": 768,
        "bg": 0.1,  # corresponding to 50 cd/m2 approx
    }
elif "viewpixx" in gethostname():
    SETUP = {
        "graphics": "viewpixx",
        "inputs": "responsepixx",
        "scrn": 1,
        "lut": "lut_viewpixx.csv",
        "fs": True,
        "wdth": 1920,
        "hght": 1080,
        "bg": 0.27,  # corresponding to 50 cd/m2 approx
    }
else:
    SETUP = {
        "graphics": "gpu",
        "inputs": "keyboard",
        "scrn": 0,
        "lut": None,
        "fs": True,
        "wdth": 1920,
        "hght": 1080,
        "bg": 0.5,
    }


def run_trials(ihrl):

    # loop over trials in block
    for trial_id in range(NTRIALS):
        
        print(f"TRIAL {trial_id}")

        # current trial variables
        trial = {}
        trial['idx'] = trial_id
        
        # SF and Contrast come from quickCSF
        contrast, sf = qcsf.next_stimulus()
        trial['contrast'] = contrast
        trial['sf'] = sf
        
        # Orientation and Phase should be randomized
        trial['phase'] = random.choice(phases)
        trial['orientation']= random.choice(orientations)
        

        # run trial
        t1 = pd.Timestamp.now().strftime("%Y%m%d:%H%M%S.%f")
        trial_results = experiment_logic.run_trial(ihrl, **trial)
        trial.update(trial_results)
        t2 = pd.Timestamp.now().strftime("%Y%m%d:%H%M%S.%f")

        # Record timing
        trial["start_time"] = t1
        trial["stop_time"] = t2

        # Save trial
        data_management.save_trial(trial, block_id=SUFFIX)
        
        # Update quickCSF
        qcsf.add_response(trial['response'])
                        

    print(f"All trials completed.")
    
    ### gets parameter estimates
    estimate = qcsf.get_estimates()

    # prints the final CSF estimates to the console
    print("CSF estimates: ", estimate)

    # saves final CSF and CIs
    filepath = data_management.results_filepath_root(block_id=SUFFIX)   
    qcsf.save_results(str(filepath))
    
    

def experiment_main(ihrl):
    # Run
    try:
        # Display instructions and wait to start
        experiment_logic.display_instructions(ihrl)
        btn, _ = ihrl.inputs.readButton(btns=["Space", "Escape", "Up", "Down", "Left", "Right"])
        if btn == "Escape":
            sys.exit("Participant terminated experiment.")

        run_trials(ihrl)

    except SystemExit as e:
        # Cleanup
        print("Exiting...")
        ihrl.close()
        raise e

    # Close session
    ihrl.close()
    print("Session complete")


if __name__ == "__main__":
    # Sound to be played at each interval
    #f1, f2, f3, f4 = 500, 600, 800, 300
    #pygame.mixer.pre_init(44100, -16, 1)

    # Create HRL interface object with parameters that depend on the setup
    ihrl = HRL(
        **SETUP,
        photometer=None,
        db=True,
    )
    
    # initializing sounds
    #sound1 = pygame.sndarray.make_sound(make_sound(f1))
    #sound2 = pygame.sndarray.make_sound(make_sound(f2))
    #sound3 = pygame.sndarray.make_sound(make_sound(f3))
    #sound4 = pygame.sndarray.make_sound(make_sound(f4))
    #sounds = [sound1, sound2, sound3, sound4]

    #ihrl.sounds = sounds
    
    experiment_main(ihrl)

    ihrl.close()
