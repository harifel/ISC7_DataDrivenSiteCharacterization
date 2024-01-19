# Data-driven site characterization for numerical modelling

This repository contains code which is used in the conference paper of the [7th International Conference on Geotechnical and Geophysical Site Characterization](https://isc7.cimne.com/). In this contribution, machine learning models were trained to obtain shear wave velocity estimates based on in-situ tests. 

The conference paper can be found here: 

## Folder structure

```
DataDriven
├── data                                  - data
├── graphics                              - saved graphics from running scripts in src
├── src                                   - folder that contains the python script files
│   ├── main.py                           - main script for now
├── environment.txt                       - dependency file to use with python
├── LICENSE                               - Github license file to specify the license of the repository 
├── README.md                             - repository description
```

## Requirements

The environment is set up using `python`.

To do this create an environment called `venv` using `environment.txt` with the help of `conda`. If you get pip errors, install pip libraries manually, e.g. `pip install pandas`
```bash
C:\Users\haris\Documents\GitHub\DATA-DRIVEN-SITE-CHARACTERIZATION>C:\Users\haris\AppData\Local\Programs\Python\Python311\python -m venv DataDriven
```

Activate the new environment with:

```bash
DataDriven\Scripts\activate
```
