# coding:utf-8
import warnings
warnings.filterwarnings('ignore')           # 警告文の無視
import re
import numpy as np
import time
import collections
import sys
sys.path.append('../..')  # 親ディレクトリのファイルをインポートするための設定
from lib import lists
import os
from key_direct import get_key

category_list = lists.category_list
categorys_list = [lists.verb_list, lists.person_name_list+["male", "female"]+lists.item_name_list,
                 lists.connector_list,
                 lists.prep_list, lists.location_name_list,
                 lists.room_name_list, lists.person_name_list+["male", "female", "place"],
                 lists.prep_list, lists.location_name_list,
                 lists.room_name_list, lists.talk_list, lists.find_type, lists.question_list,
                 lists.gesture_person_list, lists.pose_person_list, lists.object_comp_list,
                 lists.color_list, lists.cloth_list, lists.room_name_list]
data = {}
increase_data = {}
read_file_name = "command.txt"
write_file_name = "dataset.txt"

cmd_type = "create" # create or fix

LABEL_SIZE = len(categorys_list)

ENT = 10
ESC = 27
END = '\033[0m'
colors = {'BLACK': '\033[30m',
          'RED': '\033[31m',
          'GREEN': '\033[32m',
          'YELLOW': '\033[33m',
          'BLUE': '\033[34m',
          'PURPLE': '\033[35m',
          'CYAN': '\033[36m',
          'WHITE': '\033[37m',
          'BOLD': '\033[38m'}

def setting_label(text, label):
    label_list = ['<none>']*LABEL_SIZE
    label_list[0:len(label)] = label
    px = 0
    py = -1
    while True:
        cs = ""
        ls = ""
        for i, (cw, lw) in enumerate(zip(category_list, label_list)):
            if px == i:
                cw = colors['BLUE'] + cw + END
                lw = colors['GREEN'] + lw + END
            cs += cw + ' '
            ls += lw + ' '
        print("\r"+cs, end="")
        print(' '*180+'\033[180D', end="")
        print("\r"+ls, end="")
        x = ord(get_key())
        #print(x)
        if x == ENT:
            increase_data[text.replace('\t', '\n')] = ' '.join(label_list)
            break
        if x == ESC:
            x = ord(get_key())
            if x == ord('['):
                x = ord(get_key())
                if x == ord('A'):
                    # Up
                    print("up   \033[5D\033[2A", end="")
                    if py < len(categorys_list[px])-1:
                        py += 1
                elif x == ord('B'):
                    # Down
                    print("down \033[5D\033[2A", end="")
                    if py > -1:
                        py -= 1
                elif x == ord('C'):
                    # Right
                    print("right\033[5D\033[2A", end="")
                    if px < LABEL_SIZE-1:
                        px += 1
                        try:
                            py = categorys_list[px].index(label_list[px])
                        except:
                            py = -1
                elif x == ord('D'):
                    # Left    
                    print("left \033[5D\033[2A", end="")
                    if px > 0:
                        px -= 1
                        try:
                            py = categorys_list[px].index(label_list[px])
                        except:
                            py = -1
                if py == -1:
                    label_list[px] = '<none>'
                else:
                    label_list[px] = categorys_list[px][py]
        else :
            print("\033[5D\033[2A", end="")
            #print(len(x))


if __name__=='__main__':
    offset = 1
    cnt = 0
    file_path = "../data/" + write_file_name
    if not os.path.exists(file_path):
        with open(file_path, mode='w') as f:
            print(f"ファイル '{write_file_name}' が見つからなかったため、新しいファイルを生成しました。")
    with open(file_path, mode='r') as f:
        l = f.readlines()
        offset = len(l) + 1
    with open("../data/"+read_file_name) as f:
        try:
            while True:
                s_line = f.readline()
                cnt += 1
                if not s_line:
                    break
                if cnt < offset:
                    continue
                if cmd_type == "fix":
                    text, label = s_line.split('_ ')
                elif cmd_type == "create":
                    text = s_line.replace("\n", " \n")
                    label = "<none> "* LABEL_SIZE
                os.system('clear')
                print(str(cnt)+':'+text)
                setting_label(text, label.split())
            #print(data)
            #print([k for k, v in collections.Counter(label).items() if v > 0])
        except KeyboardInterrupt:
            print("--- annotation was interrupted ---")
    
    with open("../data/"+write_file_name, mode='a') as f:
        for text in increase_data:
            f.write("{}_ {}\n".format(text.replace("\n", "\t"), increase_data[text]))
    