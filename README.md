# UrbanLandUse
Characterizing urban land use with machine learning

## Summary
This repository contains a comprehensive set of instructions for creating and applying models that characterize land use / land cover (LULC) in urban areas using machine learning. The context and motivation for the project are described in detail in WRI technical note "Spatial Characterization of Urban Land Use through Machine Learning" (forthcoming in Q1 2020). The code presented here belongs to the revised and expanded methodology described in an addendum to that technical note (also forthcoming Q1 2020).  

The core workflow is encapsulated within and best understood via a sequence of Jupyter notebooks. These notebooks import and utilize a number of accompanying modules, which are simple `.py` files stored in the `utils` folder. There is also one precursor step for processing the ground-truth data from the Atlas of Urban Expansion (AUE); this was executed in QGIS via manual interaction in concert with a sequence of short Python scripts.

## Requirements
The libraries and packages required to execute the notebooks are listed in the imports block at the beginning of each. In general, these are standard geospatial and data analysis Python libraries.  

Several parts of the workflow utilize the `descarteslabs` package for imagery retreival or geospatial tiling. This is the Python API provided by Descartes Labs that provides access to its "data refinery" capabilities. Utilizing the API requires registration and token generation, as described in [their documentation](https://docs.descarteslabs.com/). Unaffiliated users may not have access to all offerings, such as certain remote sensing products. 

## Environment
Nearly all workflows, including data preparation, model training, and performance assessment, were implemented in Jupyter notebooks running a Python 3.7 kernel within a custom conda environment, and executed within those notebooks or standalone Python scripts. The computing environment was Debian (Linux) within a virtual machine hosted on Google Compute Engine, built from a Google disk image for machine learning. Actual modeling was conducted using the Keras library on top of TensorFlow. Training and model application utilized a single Tesla K80 GPU.  

Utilizing a self-contained conda environment can help avoid versioning complications and compatibility problems between various libraries. To replicate precisely the conda environment used to develop this codebase, [create an environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) using the provided [`ulu_environment.yml`](config/ulu_environment.yml) file. 

## Workflow
### 1.	Prepare Atlas of Urban Expansion files (executed in QGIS)  
-	Step-by-step [instructions](aue-preprocessing/aue-preprocessing_instructions.docx) (Word document)  
-	Helper [PyQGIS scripts](aue-preprocessing)  
- Note that preprocessing instructions and scripts are written for QGIS 2.x (2.18.13 recommended), and may not work with QGIS 3.x.
-	For a given city in the [Atlas of Urban Expansion (AUE)](http://www.atlasofurbanexpansion.org/data), create the critical geospatial file that contains all essential LULC information. Ingest and integrate the information encoded across a number of separate AUE files into a single, unified GeoJSON archive, using a combination of manual interaction and scripted processing routines.  
### 2.	Create ground-truth by rasterizing AUE vector data  
-	Notebook [core_prepare-ground-truth.ipynb](archive/phase_iv/final/core_prepare-ground-truth.ipynb)  
-	Transform vector data from the AUE into raster data amenable to machine learning. The essential input is the "complete" GeoJSON archive of the AUE data for a single city, as created in the previous step. The key output is a set of square, single-band, geospatial raster tiles, which collectively cover the entire AUE-defined study area for the city. The value of each pixel in these tiles represents the predominant LULC category at the corresponding location (or a "no data" value where that classification is unknown). These rasters effectively constitute the ground-truth data.  
### 3.	Download satellite imagery  
-	Notebook [core_acquire-imagery.ipynb](archive/phase_iv/final/core_acquire-imagery.ipynb)  
-	Specify the desired satellite imagery—from where, from when, including what spectral bands—and store it locally as multi-band, geospatial raster files. Along with the ground-truth files, these rasters form the basis of the input data for the models. Depending on the desired set of input bands, imagery from multiple sources can be acquired in a way permitting subsequent combination, although this is not explicitly demonstrated in the current codebase.  
### 4.	Construct image "chips" by fusing ground-truth and imagery  
-	Notebook [dev_building-samples_multiprocessing.ipynb](archive/phase_iv/final/dev_building-samples_multiprocessing.ipynb)  
- Extract small images ("chips") from the downloaded satellite imagery. From each downloaded image of a city, create one chip for each pixel where ground-truth is available. For every such chip, make an entry in a master catalog. These catalog entries, which include both the file path to an image chip and the corresponding LULC category of the central pixel of that chip, allow users to construct arbitrary sets of samples, whether for training, validation, or other evaluation.
- This notebook utilizes multiprocessing to speed up execution. This draft notebook directly replicates multiprocessing code from the [MPROC repository](https://github.com/brookisme/mproc), but a proper implementation would install the module and then simply load it within the notebook.
### 5.  Designate training and validation locales
- Notebook [core_apportion-locales.ipynb](archive/phase_iv/final/core_apportion-locales.ipynb)
- In order to ensure a strict separation between training and validation samples, assign every locale in every city to one of the two groups. All subsequent model training and validation can then draw samples only from the appropriate locales. 
### 6.	Create and train model
-	Notebook [core_train-model.ipynb](archive/phase_iv/final/core_train-model.ipynb)  
- Train a new model from scratch. Declare which samples are to be used for training and validation by filtering the master catalog of chips down to a desired subset. Also specify the type of classifier: a binary roads model, a full 6-category areal model, a simpler 3-category areal model which aggregates all residential categories, etc. Calculate performance statistics by applying the model to validation samples. Save all model-related objects to file, and record training parameters and model performance.
### 6.	Apply trained model to score performance  
-	Notebook [core_apply-model_scoring.ipynb](archive/phase_iv/final/core_apply-model_scoring.ipynb)  
-	Load and apply previously trained model to arbitrary set of samples, as specified by a filtered chip catalog. More precisely, apply the model to each set of samples in a list, and calculate and record performance statistics independently for each set.
### 7.  Apply trained model to generate map
- Notebook [core_apply-model_mapping.ipynb](archive/phase_iv/final/core_apply-model_mapping.ipynb)  
- Load and apply previously trained model directly to imagery. This application may include multiple, potentially overlapping, satellite captures. One at a time, the model classifies the entire area of each image that falls within the specified geospatial extent. Model output is stored in small, square tiles; these can subsequently be combined into a single LULC map (per input image).
#### Auxiliary: Inspect model output
- Notebook [helper_inspect-model-output.ipynb](archive/phase_iv/final/helper_inspect-model-output.ipynb)
- Directly inspect model output within a notebook, without needing to load an image file within dedicated GIS software.
### 8. Apply trained model to generate maps _at scale_ (external)
- Applying a trained model in order to map large areas is an important capability of the project. However, deploying mapping tasks to the Descartes Labs cloud infrastructure falls outside the scope of this repository. Instead, it is treated in [ulu_pixelwise_tasks](https://github.com/wri/ulu_pixelwise_tasks), which also relies on [dl_jobs](https://github.com/wri/dl_jobs).
- These repositories cover on-the-fly application of a trained model to imagery to generate maps. Additionally, these direct model outputs can be combined into a composite, "mode" product.
#### Auxiliary: Score performance of map product
- Notebook [dev_score-composite.ipynb](archive/phase_iv/final/dev_score-composite.ipynb)
- Compare map product within the Descartes Labs Catalog to ground-truth rasters stored locally, in order to calculate and record performance statistics.
