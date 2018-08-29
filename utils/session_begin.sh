# script for starting typical session

# should all of this happen within tmux?

# move to correct directory
cd ~/UrbanLandUse

# mount data drive
export DEV=/dev/sdb
export MNT=/data
sudo mount $DEV $MNT

# launch jupyter server
jupyter notebook --certfile=/home/Peter.Kerins/mycert.pem --keyfile /home/Peter.Kerins/mykey.key --ip=0.0.0.0 --port=8888 --no-browser &
