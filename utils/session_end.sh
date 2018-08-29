# script for ending typical session

# move to top directory
cd /

# stop jupyter server
# this no longer works -> jupyter notebook stop
pkill jupyter

# wait for process to end before trying to unmount drive
sleep 10s

# unmount data drive
sudo umount /data
