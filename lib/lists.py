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
    "answer",
    "ask",
    "bring",
    "find_object",
    "find_person",
    "follow",
    "guide",
    "manipulate",
    "introduce",
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
    # "operator",
    # "Alex",
    "Adel",
    "Angel",
    "Axel",
    "Charlie",
    "Jane",
    "Jules",
    "Morgan",
    "Paris",
    "Robin",
    "Simone",
    "James",
    # "Elizabeth",
    # "Francis",
    # "Hayden",
    # "James",
    # "Jamie",
    # "Jordan",
    # "John",
    # "Jennifer",
    # "Mary",
    # "Michael",
    # "Morgan",
    # "Patricia",
    # "Peyton",
    # "Robert",
    # "Robin",
    # "Skyler",
    # "Taylor",
    # "Tracy",
    "everyone",
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
    "bubble_tea",
    "block",
    "corn_flakes", #"cereal", "flakes", "corn flakes"
    "chips", #"potato chips",
    "chocoflake", #"choco flakes", "choco cereal", "choco cereals", "flakes of choco",
    "chocolate",
    # "chocolate_drink",
    "cloth",
    "cleaning_supply",#"cleaning stuff","cleaning supply","cleaning supplies"
    "cloth",
    "cleaning_supply",
    "cleaner",
    "crackers",
    "coke", #"cokes",
    "cookies",
    "coffee_jar",
    "cup",
    "cutlery",
    "dish", #"dishes",
    "dice",
    "detergent",
    "drink",
    "fork",
    # "frute",
    "fruit",
    "glass",
    "key",
    "knife",
    "ice_tea",
    "lunch_box",
    "light_bulb",
    "melon",
    "milk",
    "mug",
    "mustard",
    "napkin",
    "noodle", #"noodles",
    "orange",
    # "potato_chips",
    "pasta", #"spaghetti",
    "pantry_items",
    "peach",
    "pear",
    "pickles", #"pickle",
    "pocky",
    "pringles",
    "shampoo",
    "snacks",
    "strawberry",
    "soap",
    "sponge",
    "spoon",
    "sugar",
    "tea",
    "teaspoon", #"tea spoon", "tea spoons", "spoon",
    "trash", #"garbage", "debris"
    "tray",
    "toothpaste",
    "towel",
    "tonic",
    "tuna", #"tuna fish",
    "water",
    "object", #"objects", "item", "items",
]

furniture_names = [
    "bar",
    "bed",
    "bookshelf", #"shelf","bookcase"
    "bowl",
    "bin",
    "cab",#"uber"
    "cabinet",
    "chair", #"armchair", "baby chair",
    "couch", #"TV couch",
    "counter",
    "cupboard",
    # "cutlery drawer",
    "desk",
    "dining_table",
    "dishwasher", #"dish washer", "washer",
    "dining_table",
    "dinner_table",
    "drawer", #"cutlery drawer",
    "dresser",
    "end_table",
    "fireplace",
    "fridge", #"freezer",
    "house_plant",
    "kitchen_shelf",
    "kitchen_bin",
    "microwave",
    "nightstand",
    "pantry",
    "show_rack",
    "shower",
    "sink",
    "sofa",
    "stove",
    "shelf",
    "side_table",
    "office_shelf",
    "small_shelf",
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
    "living_room",
    "office"
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
    "gesture",
    "room_F"
]

what_you_say = [
    "something_about_yourself",
    "the_time",
    "what_day_is_today",
    "what_day_is_tomorrow",
    "your_team_name",
    "your_team_country",
    "your_team_affiliation",
    "the_day_of_the_week",
    "the_day_of_the_month",
    "the_day_of_the_week_tomorrow",
    "the_day_of_the_month_tomorrow",
    "a_joke",
    # "your_name",
    # "your_age",
    # "your_country",
    # "count",
    # "affiliation",
    "to_leave",
    # "question",
    # "awnser"
]

find_type = [
    "name",
    "count",
    "gender",
    "where",
    "pose"
]

superlatives = [
    "largest", #"biggest"
    "smallest", #"littlest"
    "heaviest",
    "lightest",
    "thinnest", #"flimsiest", "skinniest", "narrowest"
    # "most",
    "rightmost",
    "right",
    "left",
    "leftmost",
    "inside",
    "under",
    "over",
    "front",
    "behind"
]

quantity = [
    "2",
    "3"
]
gestures = [
    "waving",
    "raising_left_arm",
    "raising_right_arm",
    "raising_hands",
    "pointing",
    "pointing_left",
    "pointing_right",
    "sitting",
    "standing",
    "lying_down"
]