"""Manage data (design and results) for experiments
"""
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

LANG = "es"
if LANG == "de":
    participant = (input("Bitte geben Sie Ihre Initialen ein (z.b.: DEMO): ") or "DEMO").upper()
if LANG == "en":
    participant = (input("Please enter participant initials (ex.: DEMO): ") or "DEMO").upper()
if LANG == "es":
    participant = (input("Ingrese las iniciales del participante (ej.: DEMO): ") or "DEMO").upper()

# Experiment path:
experiment_path = Path().absolute()

# Overall datapath
datapath = experiment_path.parent / "data"
datapath.mkdir(parents=True, exist_ok=True)  # create datapath + parents, if does not exist
print(f"Saving and loading data in {datapath}")

# Results
results_dir = datapath / "results" / participant
results_dir.mkdir(parents=True, exist_ok=True)

# Current session (today's date)
session_id = datetime.today().strftime("%Y%m%d")


def results_filepath(block_id):
    """Construct filepath to results file for given block

    Will generally be in the form of "<results_dir>/<participant>_<session_id>_<block_id>.results.csv"

    Parameters
    ----------
    block_id : str
        identifier-string for block

    Returns
    -------
    Path
        filepath to block results file
    """

    # Results filename for this block
    filename = f"{participant}_{session_id}_{block_id}.results.csv"

    # Full filepath resultsfile
    return results_dir / filename

def results_filepath_root(block_id):
	
	filename = f"{participant}_{session_id}_{block_id}"
	
	return results_dir / filename

def save_trial(trial, block_id):
    """Save (append) whole block data to results.csv file

    Parameters
    ----------
    trial : dict
        trial data structure
    block_id : str
        string-identifier for this block
    """

    # Get filepath
    filepath = results_filepath(block_id)

    # Create, if it does not exist
    if not filepath.exists():
        print(f"creating results file {filepath}")
        with filepath.open(mode="w") as results_file:
            header_writer = csv.writer(results_file)
            header_writer.writerow(trial.keys())

    # Save
    print(f"saving trial to {filepath}")
    with filepath.open(mode="r") as results_file:
        reader = csv.DictReader(results_file)
        headers = reader.fieldnames
    with filepath.open(mode="a") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=headers)
        print(trial)
        writer.writerow(trial)


