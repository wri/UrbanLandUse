# these commands are to be run on the client/remote computer
# (ie NOT the virtual machine)

# want a pure code solution to disconnect from and stop virtual machine

# stop vm
gcloud compute instances stop "bronco-03-gpu" --project "bronco-gfw" --zone "us-central1-c"
