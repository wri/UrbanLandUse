## setup data drive - only run once
sudo bash
export DEV=/dev/sdb
mkfs.ext4 $DEV
export MNT=/data
mkdir $MNT
mount $DEV $MNT
df -h
mkdir $MNT/steven
chown -R steven:steven $MNT/steven/
exit
