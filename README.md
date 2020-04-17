# UrbanLandUse
Characterizing urban land use with machine learning

## Summary
This repository contains a comprehensive set of instructions for creating and applying models that characterize land use / land cover (LULC) in urban areas using machine learning. The context, motivation, and workflow are described in detail in WRI technical note "Spatial Characterization of Urban Land Use through Machine Learning" (forthcoming in Q1 2020).
The workflow is encapsulated within and best understood via a sequence of five Jupyter notebooks. These notebooks import and utilize a number of helper modules, which are simple `.py` files stored in the `utils` folder. There is also one precursor step for processing the ground-truth data from the Atlas of Urban Expansion (AUE); this was executed in QGIS via manual interaction in concert with a sequence of short Python scripts.

## Requirements
All listed notebooks were written for Python 2.7.13. The libraries and packages required to execute the notebooks are listed in the imports block at the beginning of each. In general, these are standard geospatial and data analysis Python libraries.
Several parts of the workflow utilize the `descarteslabs` package for imagery retreival or geospatial tiling. This is the Python API provided by Descartes Labs that provides access to its "data refinery" capabilities. Utilizing the API requires registration and token generation, as described in their documentation. Unaffiliated users may not have access to all offerings, such as certain remote sensing products. 

## Workflow
### 1.	Prepare Atlas of Urban Expansion files (executed in QGIS)  
-	Step-by-step [instructions](aue-preprocessing/aue-preprocessing_instructions.docx) (Word document)  
-	Helper [PyQGIS scripts](aue-preprocessing)  
- Note that preprocessing instructions and scripts are written for QGIS 2.x (2.18.13 recommended), and may not work with QGIS 3.x.
-	For a given city in the AUE, create the critical geospatial file that contains all essential LULC information. Ingest and integrate the information encoded across a number of separate AUE files into a single, unified geojson archive, using a combination of manual interaction and scripted processing routines.  
### 2.	Create ground-truth by rasterizing AUE vector data  
-	Notebook [core_prepare-ground-truth.ipynb](notebooks/core_prepare-ground-truth.ipynb)  
-	Transform vector data from the AUE into raster data amenable to machine learning. The essential input is the "complete" geojson archive of the AUE data for a single city, as created in the previous step. The key output is a set of square, single-band, geospatial raster tiles, which collectively cover the entire AUE-defined study area for the city. The value of each pixel in these tiles represents the predominant LULC category at the corresponding location (or a "no data" value where that classification is unknown). These rasters effectively constitute the ground-truth data.  
### 3.	Download satellite imagery  
-	Notebook [core_imagery-acquisition.ipynb](notebooks/core_imagery-acquisition.ipynb)  
-	Specify the desired satellite imagery—from where, from when, including what spectral bands—and store it locally as multi-band, geospatial raster files. These rasters are essentially the input data for the models. Depending on the desired set of input bands, imagery from multiple sources can be acquired in a way permitting subsequent combination.  
### 4.	Construct training samples by fusing ground-truth and imagery  
-	Notebook [core_build-training-data.ipynb](notebooks/core_build-training-data.ipynb)  
-	Combine the rasterized ground-truth data with coterminous imagery rasters to generate training samples: pairs of imagery input and corresponding LULC output, represented numerically and stored within NumPy arrays. These samples are automatically divided into training and validation tranches.  
### 5.	Create and train model, with 3-category model and roads model treated separately   
-	Notebook [core_train-model-3category.ipynb](notebooks/core_train-model-3category.ipynb)  
-	Notebook [core_train-model-1vsAll.ipynb](notebooks/core_train-model-1vAll.ipynb)  
-	Load selected samples—potentially from multiple images and/or multiple cities—from file into memory, and combine them into unified training and validation tranches. Create a new model and train it using the loaded samples. Calculate and store model performance statistics.   
### 6.	Apply trained model to score performance and generate maps, with the two model types again receiving slightly different treatment  
-	Notebook [core_apply-model-3category.ipynb](notebooks/core_apply-model-3category.ipynb)  
-	Notebook [core_apply-model-1vsAll.ipynb](notebooks/core_apply-model-1vAll.ipynb)  
-	Load and apply previously trained model. The model can be applied to pregenerated training and/or validation data to statistically characterize performance. Alternatively the model can be applied to imagery to classify the LULC of each pixel—including places where the actual LULC category is unknown—in order to generate comprehensive LULC maps.   

#### _release branch base commit:_
_Name: "merged"_  
_Date: 24 Dec 2018_  
_Tree: 0fbbfef878660ac650aeb26d32d0d3f2bdbbfb80_  
