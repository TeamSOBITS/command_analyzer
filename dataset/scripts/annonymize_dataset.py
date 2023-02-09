# coding:utf-8
import warnings
warnings.filterwarnings('ignore')           # 警告文の無視
import collections
import copy
import sys
sys.path.append('../..')  # 親ディレクトリのファイルをインポートするための設定
from lib import lists, dicts

read_file_name = "dataset.txt"
write_file_name = "annonymized_dataset.txt"

person_names = lists.person_names
location_names = lists.location_names
wys_lists = lists.what_you_say

gender_dict =dicts.gender_dict
item_names_dict = dicts.item_names_dict
furniture_names_dict = dicts.furniture_names_dict
room_names_dict = dicts.room_names_dict

data = {}
increase_data = {}

with open("../data/"+read_file_name) as f:
    while True:
        s_line = f.readline()
        if not s_line:
            break
        else :
            text, label = s_line.split('_ ')
            data[text.replace('\t', '\n').lower()] = label
    print("data:", len(data))
    #print(data)
    #print([k for k, v in collections.Counter(l).items() if v > 0])

category_list = ["task", "target", "prep_T1", "location_T1", "prep_T2", "location_T2", "room_T",
                   "destination", "prep_D1", "location_D1", "prep_D2", "location_D2", "room_D",
                   "WYS", "FIND", "obj_option", "obj_num", "guesture", "room_F"]

for j, text in enumerate(data):
    #print(i, d)
    new_text = text
    label = data[text]
    label_list = label.split(' ')
    new_label_list = copy.copy(label_list)
    for i, lbl in enumerate(label_list):
        if i in [0, 2, 4, 8, 10, 14]:
            continue
        #annonymize
        if lbl in item_names_dict:
            for item in item_names_dict[lbl]:
                if item == "dish" and "washer" in text:
                    continue
                elif item == "towel" and "rack" in text:
                    continue
                elif item == "spoon" and ("tea spoon" in text or "teaspoon" in text):
                    if "tea spoon" in text:
                        new_item = "tea spoon"
                    elif "teaspoon" in text:
                        new_item = "teaspoon"
                elif item == "cereal" and "choco cereal" in text:
                    new_item = "choco cereal"
                elif item == "tuna" and "tuna fish" in text:
                    new_item = "tuna fish"
                elif item == "bin" and "robin" in text:
                    continue
                elif item == "bowl" and i != 1:
                    continue
                else:
                    new_item = item
                
                if new_item+'es' in text:
                    new_tag = "<item>"
                    new_text = new_text.replace(new_item+'es', new_tag)
                    new_label_list[i] = new_tag

                elif new_item+'s' in text:
                    new_tag = "<item>"
                    new_text = new_text.replace(new_item+'s', new_tag)
                    new_label_list[i] = new_tag

                elif new_item in text:
                    new_tag = "<item>"
                    new_text = new_text.replace(new_item, new_tag)
                    new_label_list[i] = new_tag


        #名前の変換
        if lbl in person_names:
            if lbl == "person" and "someone" in text:
                new_person = "someone"
                new_tag = "<person>"
                new_text = new_text.replace(new_person, new_tag)
                new_label_list[i] = new_tag
            elif lbl == "operator" and " me" in text:
                for w in [".", ",", "?", " "]:
                    if " me"+w in text:
                        break
                new_person = "me"
                new_tag = "<person>"
                new_text = new_text.replace(" "+new_person+w, " "+new_tag+w)
                new_label_list[i] = new_tag
            elif lbl in text:
                if lbl == "person" and ("a person" in text or "the person" in text):
                    if "a person" in text:
                        new_person = "a person"
                    elif "the person" in text:
                        new_person = "the person"
                else:
                    new_person = lbl
                new_tag = "<person>"
                new_text = new_text.replace(new_person, new_tag)
                new_label_list[i] = new_tag

            elif lbl.lower() in text.lower():
                new_person = lbl
                new_tag = "<person>"
                new_text = new_text.replace(new_person.lower(), new_tag)
                new_label_list[i] = new_tag

        #性別の変換
        elif lbl in gender_dict:
            for gen in gender_dict[lbl]:
                if gen == "man" and "many" in text:
                    continue
                if gen+"s" in text:
                    continue
                if gen in text:
                    new_tag = "<gender>"
                    new_text = new_text.replace(gen, new_tag)
                    new_label_list[i] = new_tag
        
        #家具の変換
        elif lbl in furniture_names_dict:
            for furniture in furniture_names_dict[lbl]:
                if furniture == "tv" and "tv couch" in text:
                    continue
                if furniture == "shelf" and "bookshelf" in text:
                    new_furniture = "bookshelf"
                elif furniture == "chair" and ("armchair" in text or "baby chair" in text):
                    if "armchair" in text:
                        new_furniture = "armchair"
                    elif "baby chair" in text:
                        new_furniture = "baby chair"
                elif furniture == "couch" and ("tv couch" in text or "TV couch" in text):
                    if "tv couch" in text:
                        new_furniture = "tv couch"
                    elif "TV couch" in text:
                        new_furniture = "TV couch"
                elif furniture == "washer" and ("dish washer" in text or "dishwasher" in text):
                    if "dish washer" in text:
                        new_furniture = "dish washer"
                    elif "dishwasher" in text:
                        new_furniture = "dishwasher"
                elif furniture == "drawer" and "cutlery drawer" in text:
                    new_furniture = "cutlery drawer"
                elif furniture == "rack" and "towel rack" in text:
                    new_furniture = "towel rack"
                elif furniture == "table" and ("side table" in text or "dining table" in text or "coffee table" in text or "center table" in text):
                    if "side table" in text:
                        new_furniture = "side table"
                    elif "dining table" in text:
                        new_furniture = "dining table"
                    elif "coffee table" in text:
                        new_furniture = "coffee table"
                    elif "center table" in text:
                        new_furniture = "center table"
                elif furniture == "tub" and "bathtub" in text:
                    new_furniture == "bathtub"
                elif furniture == "wash" and ("washing machine" in text or "washer" in text):
                    if "washing machine" in text:
                        new_furniture = "washing machine"
                    elif "washer" in text:
                        new_furniture = "washer"
                elif furniture == "bowl" and i == 1:
                    continue
                elif furniture == "bowl" and "bowls" in text:
                    new_furniture = "bowls"
                else:
                    new_furniture = furniture
                if new_furniture in text:
                    new_tag = "<"+category_list[i]+">"
                    new_text = new_text.replace(new_furniture, new_tag)
                    new_label_list[i] = new_tag

        #場所の変換
        elif lbl in location_names and lbl in text:
            new_tag = "<"+category_list[i]+">"
            new_text = new_text.replace(lbl, new_tag)
            new_label_list[i] = new_tag

        #部屋の変換
        elif lbl in room_names_dict:
            for room in room_names_dict[lbl]:
                if room == "dining" and "dining room" in text:
                    new_room = "dining room"
                elif room == "living" and "living room" in text:
                    new_room = "living room"
                else:
                    new_room = room
                if new_room in text:
                    new_tag = "<"+category_list[i]+">"
                    new_text = new_text.replace(new_room, new_tag)
                    new_label_list[i] = new_tag

        elif lbl in wys_lists and i == 13:
            if lbl == "date" and ("what day it is" in text or "which day it is" in text):
                if "what day it is" in text:
                    wys = "what day it is"
                elif "which day it is" in text:
                    wys = "which day it is"
            elif lbl == "name" and "your name" in text:
                wys = "your name"
            elif lbl == "joke" and "a joke" in text:
                wys = "a joke"
            elif lbl == "time" and ("the time" in text or "what time it is" in text):
                if "the time" in text:
                    wys = "the time"
                elif "what time it is" in text:
                    wys = "what time it is"
            elif lbl == "day_of_week" and ("week day" in text or "the day of the week" in text):
                if "week day" in text:
                    wys = "week day"
                elif "the day of the week" in text:
                    wys = "the day of the week"
            
            elif lbl == "name" and "chairperson" in text:
                continue
            else:
                wys = lbl
            
            if wys in text:
                new_tag = "<"+category_list[i]+">"
                new_text = new_text.replace(wys, new_tag)
                new_label_list[i] = new_tag

            
            
                    
    # if ">s" in new_text:
    #     new_text = new_text.replace(">s", ">")
    increase_data[new_text] = ' '.join(new_label_list)

    

print("annonymized_data:", len(increase_data))

with open("../data/"+write_file_name, mode='w') as f:
    for text in increase_data:
        f.write("{}_ {}".format(text.replace("\n", "\t"), increase_data[text]))