#!/bin/bash

dataset_name=""
full_dataset_filelist=""
num_iterations=""
gamma=""
epochs="100"
keep_interval="20" #keep every nth checkpoint

function print_usage(){
	help_text=$"OPTIONS\n-d name of the dataset (required)\n-f path to txt file containing dataset file listing (required)\n-n number of training runs (required)\n-g gamma(required)\n-e number of epochs per training run (defaults to 100)\n-k skipping interval for training checkpoints to keep (defaults to 20)\n"
	printf "$help_text"
}

function clean_checkpoints(){
	# gets rid of the checkpoints specified by keep_interval
	directory="$1"
	num_epochs="$2"
	interval="$3"

	num_checkpoints=`ls -1 $directory/G_* | wc -l`
	mv $directory/G_${num_checkpoints}.pth tmp
    for i in $(seq 1 $num_epochs); do
    	if [ $(( i % interval )) -ne 0 ]; then
    		rm "$directory/G_${i}.pth"
    	fi
	done
	mv tmp/G_${num_checkpoints}.pth $directory
}

while getopts "hd:f:n:g:e:k:" flag; do
	case "${flag}" in
		h) print_usage; exit ;;
		d) dataset_name="${OPTARG}" ;;
		f) full_dataset_filelist="${OPTARG}" ;;
		n) num_iterations="${OPTARG}" ;;
		g) gamma="${OPTARG}" ;;
		e) epochs="${OPTARG}" ;;
		k) keep_interval="${OPTARG}" ;;
		:) echo "Missing option argument for -$OPTARG"; exit 1;;
		*) print_usage; exit 1 ;;
	esac 
done

if [ -z $dataset_name  ]; then
	echo "Dataset name cannot be empty."
	print_usage; exit 1 
fi


if [ -z $full_dataset_filelist  ]; then
	echo "Path to filelist cannot be empty."
	print_usage; exit 1 
fi

if [ -z $num_iterations  ]; then
	echo "Number of iterations cannot be empty."
	print_usage; exit 1 
fi

if [ -z $gamma  ]; then
	echo "Gamma cannot be empty."
	print_usage; exit 1 
fi


echo "Dataset name: $dataset_name"
echo "Path:         $full_dataset_filelist"
echo "Gamma:        $gamma"
echo "No. of runs:  $num_iterations"
echo "Epochs/run:   $epochs"
echo "Keep every:   $keep_interval"

python generate_filelists.py $dataset_name $full_dataset_filelist $gamma $num_iterations $epochs | tee return_file
new_dir=`cat return_file | tail -1`
model_prefix=`echo "$new_dir" | cut -d '/' -f 2`
rm return_file

while read -r run ; do
	if [ -f  $new_dir/$run/checkpoint.complete ]; then
		echo "Model for $run has already been trained - skipping..."
	else
		echo "------------Beginning $run------------"
		echo ""
		model_name="${model_prefix}_${run}"
	    ./train_ddi.sh $new_dir/$run/config.json $model_name
	    cp train_logs/$model_name/train.log $new_dir/$run/
	    python extract_loss.py filelists/$model_prefix/$run/train.log filelists/$model_prefix/$run/$run
	    clean_checkpoints train_logs/$model_name $epochs $keep_interval
	    touch $new_dir/$run/checkpoint.complete
	    echo ""
	    echo "------------$run complete------------"
	fi
done < <(ls -1 $new_dir)