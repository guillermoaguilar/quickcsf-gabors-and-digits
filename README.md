# Quickcsf (Contrast sensitivity function) for Gabors and Digits


## Requirements

- python 3
- hrl
- stimupy
- numpy

## How to run

Navigate to either folder `digits` or `gabors`, depending on which experiment you want to run

```bash
cd gabors/experiment
```

or

```bash
cd digits/experiment
```


Then run the experiment with


```bash
python run_experiment.py
```


The script will ask you for initials. This will be used as a prefix for the stored data. The data is stored inside
`gabors/data` or `digits/data/`.
