# script for ending typical session

# move to top directory
cd /

# stop jupyter server
# this no longer works -> jupyter notebook stop
pkill jupyter

# unmount data drive
sudo umount /data
