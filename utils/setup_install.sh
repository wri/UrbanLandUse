
## one time
sudo bash

apt-get update

apt-get -y install build-essential python-dev gfortran
apt-get -y install python-numpy python-scipy python-matplotlib python-pandas
apt-get -y install libproj-dev proj-data proj-bin
apt-get -y install libgeos-dev
apt-get -y install libgdal-dev python-gdal gdal-bin
apt-get -y install unzip
apt-get -y install ffmpeg libav-tools
apt-get -y install python-pip python3-pip
apt-get -y install ipython

apt-get upgrade

exit

pip install --upgrade pip

# utilities
pip install --upgrade screen emacs pprint multiprocessing

# analysis
pip install --upgrade numpy scipy sklearn statsmodels pandas
pip install --upgrade matplotlib seaborn ggplot plotnine
pip install --upgrade cython rasterio
pip install --upgrade notebook nbconvert
pip install --upgrade geopandas OSMnx
pip install --upgrade h5py

#gis
pip install --upgrade geojson cartopy shapely fiona
pip install --upgrade descarteslabs


