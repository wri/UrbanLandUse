### for code that needs to be kept on the virtual machine or local machine
### but doesn't fit elsewhere


### VIRTUAL MACHINE
### linux

### for tracking code to be added to ~/.bashrc

export ULU_REPO=$HOME/UrbanLandUse
export JUP_PORT=8888

# call with function name, no parentheses or brackets
scorecards_vm_to_bucket(){ 
	gsutil cp /data/phase_iv/models/scorecard_phase_iv_*.csv gs://wri-bronco/transfer ; 
}

jup_start(){ 
	pushd $ULU_REPO; 
	jupyter notebook --ip=0.0.0.0 --port=$JUP_PORT --no-browser & 
	popd
}
jup_stop(){ 
	pkill jupyter ; 
}

mount_data(){
	pushd $ULU_REPO
	export DEV=/dev/sdb
	export MNT=/data
	sudo mount -o discard,defaults $DEV $MNT
	popd
}

watch_gpu(){
	watch -d -n 0.5 nvidia-smi
}

# add a file counting function based on ls * | wc -l, with param for specifying filename pattern

### LOCAL MACHINE
### windows

# call with function name, no parentheses or brackets
function scorecards_bucket_to_local{ gsutil cp gs://wri-bronco/transfer/scorecard_phase_iii_*.csv "$HOME/World Resources Institute/Urban Land Use - Documents/WRI Results/phase_iii" }