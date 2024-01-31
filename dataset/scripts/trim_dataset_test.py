# coding:utf-8
import sys
import random
from tqdm import tqdm

# データセットのサイズを任意の数にトリミングするプログラム
# トリミングサイズ
args = sys.argv
trim_size = int(args[1])
input_file_name = args[2]
output_file_name = "train_" + str(trim_size) + "_test.txt"



# ファイルを読み込んでwhileで1行ずつ見ていく
input_file = open("../data/"+input_file_name)
dataset = input_file.readlines()
input_file.close()
random.shuffle(dataset)
#output_file = open(output_file_name, "a")
with tqdm(total = trim_size, leave=False) as bar:
    with open("../data/"+output_file_name,"w") as f:
        f.write(str("text	label\n"))
        for i, line in enumerate(dataset):
            bar.update(1)
            if i >= trim_size:
                break
            f.write(str(line))

print("trimed_data : {}".format(i))
