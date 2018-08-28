# script for starting typical session

# move to correct directory
cd ~/bronco

# mount data drive
export DEV=/dev/sdb
export MNT=/data
sudo mount $DEV $MNT

# launch jupyter server
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser &
