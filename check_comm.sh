comm -12 <(sort $1) <(sort $2) > comm.txt
common_lines=`cat comm.txt | wc -l`
total_lines=`cat $1 | wc -l`
perc=`perl -E 'say $ARGV[0]*100/$ARGV[1]' "$common_lines" "$total_lines"`
printf "%.3f%% similar\n" $perc 
rm comm.txt