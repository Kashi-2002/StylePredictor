#! /usr/bin/bash
import argv

echo "To run the following program run this command in terminal with the following inputs"
sleep 0.001s
echo "path to csv of the dataset"
# sleep 0.001s
read pathtocsv


echo "whether the function should run globally or only selected feature. Select Global/Selected"

read globalornot


# argv = sys.argv[1:]


echo "Value for frequency thresholding"
read frequencythresh


echo "Value for positive probability thresholding"
read posprob


echo "Value for negative probability thresholding"
read negprob


echo "Whether to apply variance thresholding or not"
read vari


echo "Whether to apply thresholding on basis of crosses"
read cros

echo "Takes input a list of styles"

while read line
do
    my_array=("${my_array[@]}" $line)
done

echo ${my_array[@]}

python main.py --path=$pathtocsv --glob=$globalornot --style=$styles --freq=$frequencythresh --pos=$posprob --neg=$negprob --var=$vari --cross=$cros