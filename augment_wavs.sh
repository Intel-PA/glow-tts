augs=""
num_wavs=""
dataset_name=""
wav_filelist=""
pitch_range="-200;200" #semitones
speed_range="0.8;1.2"

aug_methods="pitchshift, speed"
augs_arr=()
filelist=""

function clean_dir_name(){
	#removes trailing /
	cleaned="$1"
	if [ "/" == "${cleaned: -1}" ]; then 
		cleaned="${cleaned::-1}" 
	fi 
	echo "$cleaned"
}

function get_label(){
	label=`cut -d'|' -f2 < <(echo "$1")`
	echo "$label" 
}

function get_dir(){
	dir=`cut -d'/' -f1 < <(echo "$1")`
	echo "$dir" 
}

function get_filename(){
	no_label=`cut -d'|' -f1 < <(echo "$1")`
	filename=`cut -d'/' -f2 < <(echo "$no_label")`
	echo "$filename" 
}



function print_usage(){
	help_text=$"OPTIONS\n-d name of the dataset (required)\n-w text file containing locations of wav files (required)\n-n number of new wavs to generate per augmentation method (required)\n-a list of augmentation methods to apply. Must be from: $aug_methods (required)\n"
	printf "$help_text"
}

function get_pitch(){
	min=`cut -d';' -f1 <(echo $1)`
	max=`cut -d';' -f2 <(echo $1)`
	python3 -c "import random; print(random.randint($min,$max))"
}

function get_speed(){
	min=`cut -d';' -f1 <(echo $1)`
	max=`cut -d';' -f2 <(echo $1)`
	python3 -c "import random; print(random.uniform($min,$max))"
}

function make_augs_arr(){
	readarray -td ' ' augs_arr < <(echo "$augs "); unset 'augs_arr[-1]'
}

function pitchshift(){
	filename="$1"
	iter="$2"
	dir="$3"

	outfile=`cut -d'.' -f1 < <(echo "$filename")`-pitch-$iter.`cut -d'.' -f2 < <(echo "$filename")`
	sox "$dir/$filename" "$dir/$outfile" pitch `get_pitch $pitch_range`
	echo "$outfile"
}

function speed(){
	filename="$1"
	iter="$2"
	dir="$3"
	
	outfile=`cut -d'.' -f1 < <(echo "$filename")`-speed-$iter.`cut -d'.' -f2 < <(echo "$filename")`
	sox "$dir/$filename" "$dir/$outfile" speed `get_speed $speed_range`
	echo "$outfile"
}

while getopts "ha:n:d:w:" flag; do
	case "${flag}" in
		h) print_usage; exit ;;
		a) augs="${OPTARG}" ;;
		n) num_wavs="${OPTARG}" ;;
		d) dataset_name="${OPTARG}" ;;
		w) wav_filelist="${OPTARG}" ;;
		:) echo "Missing option argument for -$OPTARG"; exit 1;;
		*) print_usage; exit 1 ;;
	esac 
done

if [ -z "$dataset_name"  ]; then
	echo "Dataset name cannot be empty."
	print_usage; exit 1 
fi


if [ -z "$wav_filelist"  ]; then
	echo "Path to filelist cannot be empty."
	print_usage; exit 1 
fi

if [ -z "$num_wavs"  ]; then
	echo "Number of iterations cannot be empty."
	print_usage; exit 1 
fi

if [ -z "$augs"  ]; then
	echo "Need to specify at least 1 augmentation method."
	print_usage; exit 1 
fi

echo "Augmentated dataset name:                 $dataset_name"
echo "Augmentations to apply:                   $augs"
echo "Number of new wav files per augmentation: $num_wavs"

make_augs_arr

curr="1"
total=`cat "$wav_filelist" | wc -l`
wav_dir=`get_dir $(cat "$wav_filelist" | head -1)`
while read -r entry ; do
	wav_file=`get_filename "$entry"`
	wav_label=`get_label "$entry"`
	filelist="$filelist$dataset_name/$wav_file|$wav_label\n"
	for aug in "${augs_arr[@]}"; do
		for i in $(seq 1 $num_wavs) ; do
			saved_file=`$aug $wav_file $i $wav_dir 2>/dev/null`
			if [ $? -eq 127 ]; then
				#command not found exit code
				echo "Augmentation method $aug not known, skipping..."
				break
			else
				filelist="$filelist$dataset_name/$saved_file|$wav_label\n"
			fi
		done
	done
	printf "Processed $curr/$total\\r"
	curr=$(( curr + 1 ))
done < <(cat "$wav_filelist")
printf "Processed $total/$total\\n"
printf "$filelist" > "$wav_dir/ljs_sox_1g000.txt"