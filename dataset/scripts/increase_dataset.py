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

person_names = lists.person_name_list
location_names = lists.location_name_list

gender_dict =dicts.gender_dict
location_place_names_dicts = dicts.location_name_dict
item_names_dict = dicts.item_name_dict
room_names_dict = dicts.room_name_dict

gesture_dict = dicts.gesture_person_dict
pose_dict = dicts.pose_person_dict
obj_comp_dict = dicts.object_comp_dict
obj_color_dict = dicts.color_dict
cloth_dict = dicts.cloth_dict

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

category_list = lists.category_list
talk_list  = lists.talk_list

with tqdm(total = len(category_list)-8, leave=False) as bar1:
    for i in range(len(category_list)):
        if i in [0, 2, 6]:
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
                # print(str(len(label_list)))
                if "<none>" not in label_list[i] and "<" in label_list[i] and ">" in label_list[i]:
                    # <item>
                    if "item" in label_list[i]:
                        increase_dicts = item_names_dict
                    # <gender>
                    elif label_list[i] == "<gender>":
                        increase_dicts = gender_dict
                    # <furniture> <location>
                    elif "location_place" in label_list[i]:
                        increase_dicts = location_place_names_dicts
                    elif "location" in label_list[i]:
                        increase_lists = location_names 
                    # <room>
                    elif "room" in label_list[i]:
                        increase_dicts = room_names_dict
                    # <person>
                    elif "person" in label_list[i]:
                        increase_lists = person_names
                    # <WYS>
                    elif label_list[i] == "<WYS>":
                        increase_lists = talk_list
                    # <gesture>
                    elif label_list[i] == "<gesture>":
                        increase_dicts = gesture_dict
                    # <pose>
                    elif label_list[i] == "<pose>":
                        increase_dicts = pose_dict
                    # <obj_comp>
                    elif label_list[i] == "<obj_comp>":
                        increase_dicts = obj_comp_dict
                    # <obj_color>
                    elif label_list[i] == "<obj_color>":
                        increase_dicts = obj_comp_dict
                    # <cloth>
                    elif label_list[i] == "<cloth>":
                        increase_dicts = cloth_dict

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
                            elif word == "what_day_is_today":
                                new_word = "what day it is today"
                                new_text = text.replace(label_list[i], new_word)
                                new_label = label.replace(label_list[i], word)
                                increase_data[new_text] = new_label
                                new_word = "which day it is today"
                            elif word == "what_day_is_tomorrow":
                                new_word = "what day it is tomorrow"
                                new_text = text.replace(label_list[i], new_word)
                                new_label = label.replace(label_list[i], word)
                                increase_data[new_text] = new_label
                                new_word = "which day it is tomorrow"
                            elif word == "the_day_of_the_week":
                                new_word = "the week day"
                                new_text = text.replace(label_list[i], new_word)
                                new_label = label.replace(label_list[i], word)
                                increase_data[new_text] = new_label
                                new_word = "the day of the week"
                            elif word == "the_day_of_the_month":
                                new_word = "the month"
                                new_text = text.replace(label_list[i], new_word)
                                new_label = label.replace(label_list[i], word)
                                increase_data[new_text] = new_label
                                new_word = "the day of the month"
                            elif word == "the_day_of_the_week_tomorrow":
                                new_word = "the week day tomorrow"
                                new_text = text.replace(label_list[i], new_word)
                                new_label = label.replace(label_list[i], word)
                                increase_data[new_text] = new_label
                                new_word = "the day of the week tomorrow"
                            elif word == "the_day_of_the_month_tomorrow":
                                new_word = "the month tomorrow"
                                new_text = text.replace(label_list[i], new_word)
                                new_label = label.replace(label_list[i], word)
                                increase_data[new_text] = new_label
                                new_word = "the day of the month tomorrow"
                            elif word == "the_time":
                                new_word = "the time"
                                new_text = text.replace(label_list[i], new_word)
                                new_label = label.replace(label_list[i], word)
                                increase_data[new_text] = new_label
                                new_word = "what time it is"
                            elif word == "something_about_yourself":
                                new_word = "something about yourself"
                                new_text = text.replace(label_list[i], new_word)
                                new_label = label.replace(label_list[i], word)
                                increase_data[new_text] = new_label
                                new_word = "yourself"

                            elif word == "your_team_name":
                                new_word = "your team name"
                            # elif word == "your_age":
                            #     new_word = "your age"
                            elif word == "your_team_country":
                                new_word = "your team country"
                            elif word == "your_team_affiliation":
                                new_word = "your team affiliation"
                            elif word == "to_leave":
                                new_word = "to leave"
                            elif word == "a_joke":
                                new_word = "a joke"
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
