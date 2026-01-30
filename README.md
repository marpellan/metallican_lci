# MetalliCan LCIs
This repository contains the code and data associated with the creation of site-specific life cycle inventories (LCIs) for metal production in Canada. 

# How it works? 
The workflow is organized into several Jupyter notebooks, each focusing on a specific step of the process. 
The main functions and classes used in the notebooks are available in the `core` folder as Python scripts.

## Regioinvent and additional LCIs import 
First, we import the Regioinvent database and additional LCIs for metal production in Canada into Brightway2. See the `0. Import_RI_and_add_LCIs.ipynb` notebook for details.
We use the [Regioinvent v1.3](https://github.com/CIRAIG/Regioinvent) package  for the creation of both spatialized biosphere data and consumption markets.
We rely on Ecoinvent 3.10 cutoff database for background data.

## Parametrization from MetalliCan data
After cleaning MetalliCan data, we seek parametrization from selected sites. This notably includes energy consumption vs ore grade and land occupation vs ore processed. 
See `2. MetalliCan_parametrization.ipynb` notebook for details.

## Data-gap filling
We fill data gaps in MetalliCan data using various approaches, including scaling from similar sites, using average values, or applying engineering estimates. 
See the `3. Data_gap_filling.ipynb` notebook for details, as well as the `data_gap_filling.py` script for the functions used.

## Normalization and allocation
Once gaps are filled, we normalize the data to a common functional unit and allocate multi-output processes using appropriate allocation methods (e.g., mass, economic value).
See the `normalization_allocation.py` script for the functions used.

## Creation of Brightway LCIs
The `lci_database_builder.py` script allows to create Brightway2 LCIs. See the `4. Initialize_site_specific_LCIs.ipynb` notebook for details.

## LCA calculations
We perform LCA calculations using Impact World+ 2.1. version to assess the environmental impacts of the site-specific LCIs. 
See the `5. LCA_calculations.ipynb` notebook for details.

# 📦 Dependencies
- Brightway2.4.7
- Regioinvent v1.3
- Ecoinvent 3.10 cutoff database

# 📄 License
This repository is licensed under the BSD 3-Clause License. See the LICENSE file for details.

# 📬 Contact
For questions, feel free to open an issue or reach out via email at: [marin.pellan@polymtl.ca](mailto:marin.pellan@polymtl.ca)

# 📄 Citation
Zenodo: https://zenodo.org/records/18428434

