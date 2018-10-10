### for code that needs to be kept on the virtual machine or local machine
### but doesn't fit elsewhere


### VIRTUAL MACHINE
### linux

### for tracking code to be added to ~/.bashrc

export ULU_REPO=$HOME/UrbanLandUse

scorecards_vm_to_bucket()	{ gsutil cp /data/phase_iii/models/scorecard_phase_iii_*.csv gs://wri-bronco/transfer }


### LOCAL MACHINE
### windows

function scorecards_bucket_to_local{ gsutil cp gs://wri-bronco/transfer/scorecard_phase_iii_*.csv "$HOME/World Resources Institute/Urban Land Use - Documents/WRI Results/phase_iii" }