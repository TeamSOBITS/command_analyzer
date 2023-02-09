# coding:utf-8
import sys
import random
from tqdm import tqdm
input_file_name = "increased_dataset.txt"
output_file_name = "train_10.txt"

# データセットのサイズを任意の数にトリミングするプログラム

# トリミングサイズ
trim_size = 10

trim_cmd = None # "bring"

# ファイルを読み込んでwhileで1行ずつ見ていく
input_file = open(input_file_name)
dataset = input_file.readlines()
input_file.close()
random.shuffle(dataset)
#output_file = open(output_file_name, "a")
with tqdm(total = trim_size, leave=False) as bar:
    with open(output_file_name,"w") as f:
        for i, line in enumerate(dataset):
            bar.update(1)
            if i >= trim_size:
                break
            if trim_cmd != None and (trim_cmd not in line or trim_cmd +" object" in line):
                trim_size += 1
                continue
            f.write(str(line))
