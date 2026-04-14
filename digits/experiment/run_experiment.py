#!/usr/bin/env python
"""
Contrast detection in a Yes-No task with DIGITS.
The contrast and frequency is adaptively adjusted using the 
quickCSF algorithm (Lesmes et al. 2010)

Uses HRL on python 3

@authors: G. Aguilar, April 2026
"""
import random
from math import pi
import pygame
import pandas as pd
import numpy as np
from socket import gethostname
from hrl import HRL

import data_management
import text_displays
import experiment_logic
from stimuli import PPD

from qCSF import qCSF

NTRIALS = 100
SUFFIX = 'DIGITS'

# Possible digits
digits = [0, 2, 3, 4, 5, 6, 7, 8, 9]


# vector of possible frequencies - Zheng et al. 2018 and 2019
frequency_vector = np.array([0.5, 1, 2, 4, 8, 15.8])
contrast_vector = np.logspace(np.log10(.001),
                              np.log10(.2),
                              25, endpoint=True)

# Initialize quickCSF object                                         
qcsf = qCSF(frequency_vector=frequency_vector,
            contrast_vector=contrast_vector)

# psychometric function parameters
# fixing upper and lower asymptotes to 4% 
qcsf.lapserate = 0.04
qcsf.guessrate = float(1/9) # as there are 9 digits

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
        "lut": 'lut.csv',
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
        
        # which digit to show chosen at random
        trial['digit'] = random.choice(digits)   

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

    # Create HRL interface object with parameters that depend on the setup
    ihrl = HRL(
        **SETUP,
        photometer=None,
        db=True,
    )
    
    experiment_main(ihrl)

    ihrl.close()
