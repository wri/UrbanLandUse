# CUDA for Ubuntu 17.04
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64/7fa2af80.pub

# CUDA 9.0
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64/cuda-repo-ubuntu1704_9.0.176-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1704_9.0.176-1_amd64.deb
rm cuda-repo-ubuntu1704_9.0.176-1_amd64.deb
sudo apt-get install cuda -y
# ... and then reboot node

nvidia-smi  ## success

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64' >> ~/.bashrc
source ~/.bashrc

gsutil cp gs://wri-bronco/data/software/cudnn-9.1-linux-x64-v7.tgz .
tar xfvz cudnn-9.1-linux-x64-v7.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn_static.a /usr/local/cuda/lib64/
sudo cp cuda/lib64/libcudnn.so.7.0.5 /usr/local/cuda/lib64/
sudo ln -s /usr/local/cuda/lib64/libcudnn.so.7.0.5 /usr/local/cuda/lib64/libcudnn.so.7
sudo ln -s /usr/local/cuda/lib64/libcudnn.so.7.0.5 /usr/local/cuda/lib64/libcudnn.so

## IBM MKL DL kernel library
sudo apt install cmake

git clone https://github.com/01org/mkl-dnn.git
cd mkl-dnn/scripts && ./prepare_mkl.sh && cd ..
mkdir -p build && cd build && cmake .. && make
sudo make install

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib' >> ~/.bashrc
source ~/.bashrc

## Google TensorFlow

# sudo pip3 install tensorflow-gpu

# python 2.7
sudo pip3 install --ignore-installed --upgrade  https://github.com/mind/wheels/releases/download/tf1.4-gpu-cuda9/tensorflow-1.4.0-cp27-cp27mu-linux_x86_64.whl

# python 3.5
sudo pip3 install --ignore-installed --upgrade  https://github.com/mind/wheels/releases/download/tf1.4-gpu-cuda9/tensorflow-1.4.0-cp35-cp35m-linux_x86_64.whl

## Keras
sudo pip3 install h5py
sudo pip3 install keras

