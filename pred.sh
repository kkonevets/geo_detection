#!/bin/bash

printf "extid %s\n\n" $(sed "$1q;d" extids.csv)

echo "geography"
counter=0
for cix in $(sed "$1q;d" city_ixs.csv)
do
    cix=${cix%$'\r'}
    if [[ "$counter" -gt 0 ]]; then
    	echo $(sed "$((${cix} + 1))q;d" cities_splited.csv)
    else
    	echo $cix
    fi
    counter=$((counter+1))
done

printf "\nprobs\n"
sed "$1q;d" probs.csv