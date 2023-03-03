# coding:utf-8
import sys
import random
from tqdm import tqdm
input_file_name = "increased_dataset.txt"
output_file_name = "train_98700.txt"

# データセットのサイズを任意の数にトリミングするプログラム

# トリミングサイズ
trim_size = 98700

# ファイルを読み込んでwhileで1行ずつ見ていく
input_file = open("../data/"+input_file_name)
dataset = input_file.readlines()
input_file.close()
random.shuffle(dataset)
#output_file = open(output_file_name, "a")
with tqdm(total = trim_size, leave=False) as bar:
    with open("../data/"+output_file_name,"w") as f:
        for i, line in enumerate(dataset):
            bar.update(1)
            if i >= trim_size:
                break
            f.write(str(line))

print("trimed_data : {}".format(i))
