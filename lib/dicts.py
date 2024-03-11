#!/usr/bin/env python3
#coding:utf-8


class dict:
    def __init__(self):
        self.itos = []
        self.stoi = {}

    def create_dict(self,list):
        self.itos = list
        for i, w in enumerate(list):
            self.stoi[w] = i

#正解ラベルの辞書
# 辞書のキー値は正解ラベルを，値はラベルの単語を指すコマンド内の単語を示す
verc_dict = {
    "take": ["take", "get", "grasp", "fetch"],
    "place": ["put", "place"],
    "deliver": ["bring", "give", "deliver"],
    "bring": ["bring", "give"],
    "go": ["go", "navigate"],
    "find": ["find", "locate", "look for"],
    "talk": ["tell", "say"],
    "answer": ["answer"],
    "meet": ["meet"],
    "tell": ["tell"],
    "greet": ["greet", "salute", "say hello to", "introduce yourself to"],
    "remember": ["meet", "contact", "get to know", "get acquainted with"],
    "count": ["tell me how many"],
    "describe": ["tell me how", "describe"],
    "offer": ["offer"],
    "follow": ["follow"],
    "guide": ["guide", "escort", "take", "lead"],
    "accompany": ["accompany"]
}

prep_dict = {
    "deliverPrep": ["to"],
    "placePrep": ["on"],
    "inLocPrep": ["in"],
    "fromLocPrep": ["from"],
    "toLocPrep": ["to"],
    "atLocPrep": ["at"],
    "talkPrep": ["to"],
    "locPrep": ["in", "at"],
    "onLocPrep": ["on"],
    "arePrep": ["are"],
    "ofPrsPrep": ["of"]
}

connector_dict = {
    "and": ["and"]
}
gesture_person_dict = {
    "waving_person": ["waving person"],
    "person_raising_their_left_arm": ["person rasing their left arm"],
    "person_raising_their_right_arm": ["person rasing their right arm"],
    "person_pointing_to_the_left": ["person pointing to the left"],
    "person_pointing_to_the_right": ["person pointing to the right"]
}

pose_person_dict = {
    "sitting_person": ["sitting person"],
    "standing_person": ["standing person"],
    "lying_person": ["lying person"]
}


object_comp_dict = {
    "biggest": ["biggest"],
    "largest": ["largest"], 
    "smallest": ["smallest"], 
    "heaviest": ["heaviest"], 
    "lightest": ["lightest"], 
    "thinnest": ["thinnest"]
}

talk_dict = {
    "something_about_yourself": ["something about yourself"], 
    "the_time": ["the time"], 
    "what_day_is_today": ["what day is today"], 
    "what_day_is_tomorrow": ["what day is tomorrow"], 
    "your_teams_name": ["your teams name"],
    "your_teams_country": ["your teams country"], 
    "your_teams_affiliation": ["your teams affiliation"], 
    "the_day_of_the_week": ["the day of the week"], 
    "the_day_of_the_month": ["the day of the month"]
}

question_dict = {
    "question": ["question"],
    "quiz": ["quiz"]
}

color_dict = {
    "blue": ["blue"], 
    "yellow": ["yellow"], 
    "black": ["black"],
    "white": ["white"], 
    "red": ["red"], 
    "orange": ["orange"], 
    "gray": ["gray"]
}

cloth_dict = {
    "t_shirt": ["t shirt", "t shirts"], 
    "shirt": ["shrit", "shrits"], 
    "blouse": ["blouse", "blouses"], 
    "sweater": ["sweater", "sweaters"], 
    "coat": ["coat", "coats"], 
    "jacket": ["jacket", "jackets"]
}

person_name_dict = {
    "Adel": ["Abel"],
    "Angel": ["Angel"],
    "Axel": ["Axel"],
    "Charlie": ["Charlie"],
    "Jane": ["Jane"],
    "Jules": ["Jules"],
    "Morgan": ["Morgan"],
    "Paris": ["Paris"],
    "Robin": ["Robin"],
    "Simone": ["Simone"]
}

gender_dict = {
    "male":["male", "males", "boy", "boys", "man", "men", "gentleman", "gentlemen",],
    "female":["female", "females", "girl", "girls", "woman", "women", "lady", "ladies"]
}

item_name_dict = {
    #drink
    "orange_juice": ["orange juice"],
    "red_wine": ["red wine"],
    "milk": ["milk"],
    "iced_tea": ["iced tea"],
    "cola": ["cola"],
    "tropical_juice": ["tropical juice"],
    "juice_pack": ["juice pack"],
    #fruit
    "apple": ["apple"],
    "pear": ["pear"],
    "lemon": ["lemon"],
    "peach": ["peach"],
    "banana": ["banana"],
    "strawberry": ["strawberry"],
    "orange": ["orange"],
    "plum": ["plum"],
    #snack
    "cheezit": ["cheezit"],
    "cornflakes": ["cornflakes"],
    "pringles": ["pringles"],
    #food
    "tuna": ["tuna"],
    "suger": ["sugar"],
    "strawberry_jello": ["strawberry jello"],
    "tomato_soup": ["tomato soup"],
    "mustard": ["mustard"],
    "chocolate_jello": ["chocolate jello"],
    "spam": ["spam"],
    "coffee_grounds": ["cofee grounds"],
    #dish
    "plate": ["plate"],
    "fork": ["fork"],
    "spoon": ["spoon"],
    "cup": ["cup"],
    "knife": ["knife"],
    "bowl": ["bowl"],
    #toy
    "rubiks_cube": ["rubiks cube"],
    "soccer_ball": ["soccer ball"],
    "dice": ["dice"],
    "tennis_ball": ["tennis ball"],
    "baseball": ["baseball"],
    #cleaning_supply
    "cleanser": ["cleaner"],
    "sponge": ["sponge"]
}

room_name_dict = {
    "bedroom": ["bedroom"],
    "kitchen": ["kitchen"],
    "office": ["office"],
    "living_room": ["living room"],
    "bathroom": ["bathroom"]
}

location_name_dict = {
    "bed": ["bed"], #p
    "bedside_table": ["bedside table"], #p
    "shelf": ["shelf"], #cleaning supplies #p
    "trashbin": ["trashbin"],
    "dishwasher": ["dishwasher"], #p
    "potted_plant": ["potted plant"], 
    "kitchen_table": ["kitchen table"], #dishes #p
    "chairs": ["chairs"], 
    "pantry": ["pantry"], #food #p
    "refrigerator": ["refrigerator"], #p
    "sink": ["sink"], #p
    "cabinet": ["cabinet"], #drinks #p
    "coatrack": ["coatrack"], 
    "desk": ["desk"], #fruits #p
    "armchair": ["armchair"],
    "desk_lamp": ["desk lamp"], 
    "waste_basket": ["waste basket"], 
    "tv_stand": ["tv stand"], #p
    "storage_rack": ["storage rack"], #p
    "lamp": ["lamp"], 
    "side_tables": ["side tables"], #snack #p
    "sofa": ["sofa"], #p
    "bookshelf": ["bookshelf"], #toys, #p
    "entrance": ["entrance"],
    "exit": ["exit"]
}

result_dict = {
    "task":"<none>",
    "target":"<none>",
    "and":"<none>",
    "prep_T":"<none>",
    "location_T":"<none>",
    "room_T":"<none>",
    "destination":"<none>",
    "prep_D":"<none>",
    "location_D":"<none>",
    "room_D":"<none>",
    "WYS":"<none>",
    "FIND":"<none>",
    "QUESTION":"<none>",
    "gesture":"<none>",
    "pose":"<none>",
    "obj_comp":"<none>",
    "obj_color":"<none>",
    "cloth":"<none>",
    "room_F":"<none>"
}

