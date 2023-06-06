#!/usr/bin/env python3
#coding:utf-8

#正解ラベルのリスト
# リスト内のコメントアウトはコマンド中には存在するが，正解ラベルは同一行内の単語に統一されたものとする
remove_words = [
    "please",
    "Please",
   ",",
    ".",
    "?",
    "\t",
    "\n"
]

verb_list = [
    "answer",
    "ask",
    "bring",
    "find",
    "follow",
    "guide",
    "grasp",
    "introduce",
    "put",
    "else"
]

prep_list = [
    "in",
    "on",
    "under",
    "above",
    "into",
    "out",
    "right",
    "left",
    "top",
    "behind",
    "near", # next
    "at"
]

person_names = [
    "operator",
    "Alex",
    "Charlie",
    "Elizabeth",
    "Francis",
    "Hayden",
    "James",
    "Jamie",
    "Jordan",
    "John",
    "Jennifer",
    "Mary",
    "Michael",
    "Morgan",
    "Patricia",
    "Peyton",
    "Robert",
    "Robin",
    "Skyler",
    "Taylor",
    "Tracy",
    "person"
]

gender = [
    "male",
    "males",
    "boy",
    "boys",
    "man",
    "men",
    "gentleman",
    "gentlemen",
    "female",
    "females",
    "girl",
    "girls",
    "woman",
    "women",
    "lady",
    "ladies"
    ]

category = [
    "food",
    "kitchen_item",
    "task_item"
    ]

item_names = [
    "apple",
    "bag",
    "banana",
    "beer", #"beers",
    "box",
    "bowl",
    "block",
    "cereal", #"flakes"
    "chips", #"potato chips",
    "chocoflake", #"choco flakes", "choco cereal", "choco cereals", "flakes of choco",
    "chocolate",
    "chocolate_drink",
    "cloth",
    "cleaning_supply",#"cleaning_stuff"
    "coke", #"cokes",
    "cookies",
    "cup",
    "cutlery",
    "dish", #"dishes",
    "dice",
    "detergent",
    "drink",
    "fork",
    "frute",
    "glass",
    "key",
    "knife",
    "lunch_box",
    "light_bulb",
    "melon",
    "milk",
    "mug",
    "napkin",
    "noodle", #"noodles",
    "potato_chips",
    "object", #"objects", "item", "items",
    "orange",
    "pasta", #"spaghetti",
    "peach",
    "pear",
    "pickles", #"pickle",
    "shampoo",
    "snacks",
    "soap",
    "sponge",
    "spoon",
    "tea",
    "teaspoon", #"tea spoon", "tea spoons", "spoon",
    "trash", #"garbage", "debris"
    "tray",
    "toothpaste",
    "towel",
    "tuna", #"tuna fish",
    "water"
]

furniture_names = [
    "bar",
    "bed",
    "bookshelf", #"shelf","bookcase"
    "bowl",
    "bin",
    "cab",
    "cabinet",
    "chair", #"armchair", "baby chair",
    "couch", #"TV couch",
    "counter",
    "cupboard",
    # "cutlery drawer",
    "desk",
    "dishwasher", #"dish washer", "washer",
    "drawer", #"cutlery drawer",
    "dresser",
    "end_table",
    "fireplace",
    "fridge", #"freezer",
    "microwave",
    "nightstand",
    "shower",
    "sink",
    "sofa",
    "stove",
    "shelf",
    "tall_table",
    "table", #"side table", "dining table", "coffee table", "center table",
    "storage_table",
    "side_table",
    "towel_rack",
    "tub", #"bathtub",
    "tv", 
    "washing_machine"
]

room_names = [
    "bathroom",
    "bedroom",
    "dining_room",
    "kitchen",
    "living_room"
]

location_names = [
    "bar",
    "corridor", #"CORRIDOR",
    "entrance",
    "hallway",
    "toilet",
    "wardrobe"
]

category_list = [
    "task",
    "target",
    "prep_T1",
    "location_T1",
    "prep_T2",
    "location_T2",
    "room_T",
    "destination",
    "prep_D1",
    "location_D1",
    "prep_D2",
    "location_D2",
    "room_D",
    "WYS",
    "FIND",
    "obj_option",
    "obj_num",
    "guesture",
    "room_F"
]

what_you_say = [
    "name",
    "age",
    "yourself",
    "country",
    "count",
    "affiliation",
    "joke",
    "time",
    "date",
    "day_of_week",
    "to_leave",
    "question"
]

find_type = [
    "name",
    "count",
    "gender",
    "where",
    "guesture"
]
superlatives = [
    "largest", #"biggest"
    "smallest", #"littlest"
    "heaviest",
    "lightest",
    "thinnest", #"flimsiest", "skinniest", "narrowest"
    "most",
    "rightmost",
    "leftmost",
]

quantity = [
    "2",
    "3"
]
guestures = [
    "waving",
    "rising_left_arm",
    "rising_right_arm",
    "rising_hands",
    "pointing",
    "pointing_left",
    "pointing_right",
    "sitting",
    "standing",
    "lying"
]