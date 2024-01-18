# MLpFEM - towards Machine Learning based parameter calibration for Finite Element Modelling

This repository contains code for the collaborative project between [NGI](https://www.ngi.no/eng) and the Graz University of Technology
[Institute of Soil Mechanics, Foundation Engineering and Computational Geotechnics](https://www.soil.tugraz.at) to explore the possibility to automatically
calibrate parameters for constitutive models for Computational Geotechnics.


The repository is currently under development.

## Folder structure

```
MLpFEM
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

To do this create an environment called `MLpFEM` using `environment.txt` with the help of `conda`. If you get pip errors, install pip libraries manually, e.g. `pip install pandas`
```bash
C:\Users\haris\Documents\GitHub\DATA-DRIVEN-SITE-CHARACTERIZATION>C:\Users\haris\AppData\Local\Programs\Python\Python311\python -m venv MLpFEM
```

Activate the new environment with:

```bash
MLpFEM\Scripts\activate
```
