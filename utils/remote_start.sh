# these commands are to be run on the client/remote computer
# (ie NOT the virtual machine)

# want a pure code solution to start and connect to virtual machine

# start vm
gcloud compute instances start "bronco-03-gpu" --project "bronco-gfw" --zone "us-central1-c"

# wait for vm to be ready to accept connections
sleep 30

# connect to vm
gcloud compute --project "bronco-gfw" ssh --zone "us-central1-c" "bronco-03-gpu"