#!/usr/bin/env python3
#coding:utf-8

#正解ラベルのリスト
# リスト内のコメントアウトはコマンド中には存在するが，正解ラベルは同一行内の単語に統一されたものとする
remove_words = [
    "please",
    "Please",
    "robot",
    "Robot",
   ",",
    ".",
    "?",
    "\t",
    "\n"
]

verb_list = [
    "take",
    "place",
    "deliver",
    "bring",
    "go",
    "find",
    "talk",
    "answer",
    "meet",
    "tell",
    "greet",
    "remember",
    "count",
    "describe",
    "offer",
    "follow",
    "guide",
    "accompany"
]

prep_list = [
    "deliverPrep",
    "placePrep",
    "inLocPrep",
    "fromLocPrep",
    "toLocPrep",
    "atLocPrep",
    "talkPrep",
    "locPrep",
    "onLocPrep",
    "arePrep",
    "ofPrsPrep"
]

connector_list = [
    "and"
]

gesture_person_list = [
    "waving_person",
    "person_raising_their_left_arm",
    "person_raising_their_right_arm",
    "person_pointing_to_the_left",
    "person_pointing_to_the_right"
]

pose_person_list = [
    "sitting_person",
    "standing_person",
    "lying_person"
]

object_comp_list = [
    "biggest",
    "largest", 
    "smallest", 
    "heaviest", 
    "lightest", 
    "thinnest"
]

talk_list = [
    "something_about_yourself", 
    "the_time", 
    "what_day_is_today", 
    "what_day_is_tomorrow", 
    "your_teams_name",
    "your_teams_country", 
    "your_teams_affiliation", 
    "the_day_of_the_week", 
    "the_day_of_the_month"
]

question_list = [
    "question",
    "quiz"
]

color_list = [
    "blue", 
    "yellow", 
    "black",
    "white", 
    "red", 
    "orange", 
    "gray"
]

cloth_list = [
    "t_shirt", 
    "shirt", 
    "blouse", 
    "sweater", 
    "coat", 
    "jacket"
]

person_name_list = [
    "Adel",
    "Angel",
    "Axel",
    "Charlie",
    "Jane",
    "Jules",
    "Morgan",
    "Paris",
    "Robin",
    "Simone"
]

gender_list = [
    "male",
    "female"
]

item_name_list = [
    #drink
    "orange_juice",
    "red_wine",
    "milk",
    "iced_tea",
    "cola",
    "tropical_juice",
    "juice_pack",
    #fruit
    "apple",
    "pear",
    "lemon",
    "peach",
    "banana",
    "strawberry",
    "orange",
    "plum",
    #snack
    "cheezit",
    "cornflakes",
    "pringles",
    #food
    "tuna",
    "suger",
    "strawberry_jello",
    "tomato_soup",
    "mustard",
    "chocolate_jello",
    "spam",
    "coffee_grounds",
    #dish
    "plate",
    "fork",
    "spoon",
    "cup",
    "knife",
    "bowl",
    #toy
    "rubiks_cube",
    "soccer_ball",
    "dice",
    "tennis_ball",
    "baseball",
    #cleaning_supply
    "cleanser",
    "sponge"
]

room_name_list = [
    "bedroom",
    "kitchen",
    "office",
    "living_room",
    "bathroom"
]

location_name_list = [
    "bed", #p
    "bedside_table", #p
    "shelf", #cleaning supplies #p
    "trashbin",
    "dishwasher", #p
    "potted_plant", 
    "kitchen_table", #dishes #p
    "chairs", 
    "pantry", #food #p
    "refrigerator", #p
    "sink", #p
    "cabinet", #drinks #p
    "coatrack", 
    "desk", #fruits #p
    "armchair",
    "desk_lamp", 
    "waste_basket", 
    "tv_stand", #p
    "storage_rack", #p
    "lamp", 
    "side_tables", #snack #p
    "sofa", #p
    "bookshelf" #toys, #p
    "entrance",
    "exit"
]

category_list = [
    "task",
    "target",
    "and",
    "prep_T",
    "location_T",
    "room_T",
    "destination",
    "prep_D",
    "location_D",
    "room_D",
    "WYS",
    "FIND",
    "QUESTION",
    "gesture",
    "pose",
    "obj_comp",
    "obj_color",
    "cloth",
    "room_F"
]

find_type = [
    "name",
    "count",
    "gender",
    "where",
    "pose"
]