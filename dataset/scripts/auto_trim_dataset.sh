#!/bin/bash

# TEXT_NUM=$((wc -l <../data/increased_dataset.txt))
# TEXT_NUM=$(wc -l <../data/$1)
TEXT_NUM=$(wc -l <../data/increased_dataset.txt)
# text_num=$(bc $TEXT_NUM / 1000)
echo 10000
# python3 trim_dataset.py $((($TEXT_NUM/1000)*1000)) $1
python3 trim_dataset.py 10000 increased_dataset.txt