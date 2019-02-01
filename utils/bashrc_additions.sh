### for code that needs to be kept on the virtual machine or local machine
### but doesn't fit elsewhere


### VIRTUAL MACHINE
### linux

### for tracking code to be added to ~/.bashrc

export ULU_REPO=$HOME/UrbanLandUse

# call with function name, no parentheses or brackets
scorecards_vm_to_bucket(){ 
	gsutil cp /data/phase_iii/models/scorecard_phase_iii_*.csv gs://wri-bronco/transfer ; 
}

jup_start(){ 
	cd $ULU_REPO; 
	jupyter notebook --certfile=/home/Peter.Kerins/mycert.pem --keyfile /home/Peter.Kerins/mykey.key --ip=0.0.0.0 --port=8888 --no-browser & 
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

# add a file counting function based on ls * | wc -l, with param for specifying filename pattern

### LOCAL MACHINE
### windows

# call with function name, no parentheses or brackets
function scorecards_bucket_to_local{ gsutil cp gs://wri-bronco/transfer/scorecard_phase_iii_*.csv "$HOME/World Resources Institute/Urban Land Use - Documents/WRI Results/phase_iii" }