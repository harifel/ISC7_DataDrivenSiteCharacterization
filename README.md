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

To do this create an environment called `venv`.
```bash
C:\Users\haris\Documents\GitHub\ISC7_DataDrivenSiteCharacterization>C:\Users\haris\AppData\Local\Programs\Python\Python311\python -m venv DataDriven
```

Activate the new environment with:
```bash
C:\Users\haris\Documents\GitHub\ISC7_DataDrivenSiteCharacterization>DataDriven\Scripts\activate
```

Install all packages using `environment.txt`. If you get pip errors, install pip libraries manually, e.g. `pip install pandas`
```bash
(venv) C:\Users\haris\Documents\GitHub\ISC7_DataDrivenSiteCharacterization>py -m pip install -r environment.txt
```

## Database for Machine Learning
The database is accessible on the website of the [Computational Geotechnics Group (Graz University of Technology)]([https://isc7.cimne.com/](https://www.tugraz.at/fileadmin/user_upload/Institute/IBG/Datenbank/Database_CPT_PremstallerGeotechnik.zip)https://www.tugraz.at/fileadmin/user_upload/Institute/IBG/Datenbank/Database_CPT_PremstallerGeotechnik.zip). A description of the database itself can be found in the paper by Oberhollenzer et al. (2019) - DOI: https://doi.org/10.1016/j.dib.2020.106618
