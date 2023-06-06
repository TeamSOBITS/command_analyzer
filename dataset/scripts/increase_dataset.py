# coding:utf-8
from itertools import count
import warnings
warnings.filterwarnings('ignore')           # 警告文の無視
import collections
import copy
import sys
sys.path.append('../..')  # 親ディレクトリのファイルをインポートするための設定
from lib import lists, dicts
from tqdm import tqdm

read_file_name = "annonymized_dataset.txt"
write_file_name = "increased_dataset.txt"

person_names = lists.person_names
location_names = lists.location_names
wys_lists = lists.what_you_say

gender_dict =dicts.gender_dict
location_place_names_dicts = dicts.location_place_names_dict
item_names_dict = dicts.item_names_dict
furniture_names_dict = dicts.furniture_names_dict
room_names_dict = dicts.room_names_dict

data = {}

with open("../data/"+read_file_name) as f:
    while True:
        s_line = f.readline()
        if not s_line:
            break
        else :
            text, label = s_line.split('_ ')
            data[text.replace('\t', '\n')] = label
            # data[text.replace('\t', '\n').lower()] = label
    print("data:", len(data))
    #print(data)
    #print([k for k, v in collections.Counter(l).items() if v > 0])

category_list = ["task", "target", "prep_T1", "location_T1", "prep_T2", "location_T2", "room_T",
                   "destination", "prep_D1", "location_D1", "prep_D2", "location_D2", "room_D",
                   "WYS", "FIND", "obj_option", "obj_num", "guesture", "room_F"]


with tqdm(total = len(category_list)-13, leave=False) as bar1:
    for i in range(len(category_list)):
        if i in [0, 2, 4, 8, 10, 14, 15, 16, 17]:
        # if i in [0, 2, 4, 5, 8, 10, 11, 12, 14, 15, 16, 17, 18]:
            continue
        bar1.update(1)
        increase_data = {}
        with tqdm(total = len(data), leave=False) as bar2:
            for j, text in enumerate(data):
                bar2.update(1)
                label = data[text]
                label_list = label.split(' ')
                increase_dicts = {}
                increase_lists =[]
                if "<none>" not in label_list[i] and "<" in label_list[i] and ">" in label_list[i]:
                    # <item>
                    if label_list[i] == "<item>":
                        increase_dicts = item_names_dict
                    # <gender>
                    elif label_list[i] == "<gender>":
                        increase_dicts = gender_dict
                    # <furniture> <location>
                    elif "location_place" in label_list[i]:
                        increase_dicts = location_place_names_dicts
                    elif "location" in label_list[i]:
                        increase_dicts = furniture_names_dict
                        increase_lists = location_names 
                    # <room>
                    elif "room" in label_list[i]:
                        increase_dicts = room_names_dict
                    # <person>
                    elif label_list[i] == "<person>":
                        increase_lists = person_names
                    # <WYS>
                    elif label_list[i] == "<WYS>":
                        increase_lists = wys_lists
                    
                    if increase_dicts != {}:
                        for words in increase_dicts:
                            for word in increase_dicts[words]:
                                new_text = text.replace(label_list[i], word)
                                new_label = label.replace(label_list[i], words)
                                increase_data[new_text] = new_label
                                
                    if increase_lists != []:
                        for word in increase_lists:
                            if word == "operator":
                                new_word = "me"
                            elif word == "date":
                                new_word = "what day it is"
                                new_text = text.replace(label_list[i], new_word)
                                new_label = label.replace(label_list[i], word)
                                increase_data[new_text] = new_label
                                new_word = "which day it is"
                            elif word == "name":
                                new_word = "your name"
                            elif word == "joke":
                                new_word = "a joke"
                            elif word == "time":
                                new_word = "the time"
                                new_text = text.replace(label_list[i], new_word)
                                new_label = label.replace(label_list[i], word)
                                increase_data[new_text] = new_label
                                new_word = "which time it is"
                            elif word == "week":
                                new_word = "week day"
                                new_text = text.replace(label_list[i], new_word)
                                new_label = label.replace(label_list[i], word)
                                increase_data[new_text] = new_label
                                new_word = "the day of the week"
                            else:
                                new_word = word
                            new_text = text.replace(label_list[i], new_word)
                            new_label = label.replace(label_list[i], word)
                            increase_data[new_text] = new_label
                    
                    if increase_dicts == {} and increase_lists == []:
                        print("---error---\n this tag is none : |{}|".format(label_list[i]))
                else:
                    increase_data[text] = label
        data = copy.copy(increase_data)


print("increase_data:", len(increase_data))

with open("../data/"+write_file_name, mode='w') as f:
    for text in increase_data:
        f.write("{}_ {}".format(text.replace("\n", "\t"), increase_data[text]))