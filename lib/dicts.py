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
    "answer":0,
    "ask":1,
    "bring":2,
    "find":3,
    "follow":4,
    "guide":5,
    "grasp":6,
    "introduce":7,
    "put":8,
    "else":9
}


replace_words = {
    " boy ":" male ",
    " boys ":" male ",
    " males ":" male ",
    " man ":" male ",
    " men ":" male ",
    " gentleman ":" male ",
    " gentlemen ":" male ",
    " girl ":" female ",
    " girls ":" female ",
    " females ":" female ",
    " woman ":" female ",
    " women ":" female ",
    " lady ":" female ",
    " ladies ":" female ",
    " chocoflakes ":" chocoflake ",
    "supplyes":"supply"
}

replace_phrases = {
    "what's":"what is",
    "who's":"who is",
    "he's":"he is",
    "there's":"there is",
    "i'll":"i will",
    "i'd":"i would",
    "choco flake":"chocoflake",
    "choco cereal":"chocoflake",
    "tea spoon":"teaspoon",
    "bookcase":"bookshelf",
    "book case":"bookshelf",
    "chocolate drink":"chocodrink",
    # "cleaning supply":"cleaning_supply",
    # "cleaning supplies":"cleaning_supply",
    # "cleaning stuff":"cleaning_supply",
    # "cleaning stuffes":"cleaning_supply",
    "dining table":"diningtable"
}


gender_dict = {
    "male":["male", "males", "boy", "boys", "man", "men", "gentleman", "gentlemen",],
    "female":["female", "females", "girl", "girls", "woman", "women", "lady", "ladies"]
    }

item_names_dict = {
    "apple":["apple"],
    "bag":["bag"],
    "banana":["banana"],
    "beer":["beer", "beers"],
    "box":["box"],
    "bowl":["bowl"],
    "block":["block", "blocks"],
    "cereal":["cereal", "flakes"],
    "chips":["chips", "potato chips"],
    "chocoflake":["choco flakes", "choco cereal", "choco cereals", "flakes of choco", "choco flake"], #"chocoflake", "chocoflakes", 
    "chocolate":["chocolate"],
    "chocodrink":["chocolate drink"],
    "cloth":["cloth"],
    "cleaning_supply":["cleaning stuffes","cleaning supply","cleaning stuff","cleaning supplies"],
    "coke":["coke", "cokes"],
    "cookies":["cookie", "cookies"],
    "cup":["cup", "cups"],
    "cutlery":["cutlery"],
    "dish":["dish", "dishes"],
    "dice":["dice", "dices"],
    "detergent":["detergent", "detergents"],
    "fork":["fork"],
    "fruit":["fruit", "fruits"],
    "frute":["frute"],
    "glass":["glass"],
    "key":["key"],
    "knife":["knife"],
    "lunch_box":["lunch box", "lunch boxes"],
    "light_bulb":["light bulb", "light bulbs"],
    "melon":["melon"],
    "milk":["milk"],
    "mug":["mug"],
    "napkin":["napkin"],
    "noodle":["noodles", "noodle"],
    "potato_chips":["potato chips"],
    "object":["object", "objects", "it", "item", "items"],
    "pasta":["pasta", "spaghetti"],
    "peach":["peach"],
    "pear":["pear"],
    "pickles":["pickles", "pickle"],
    "shampoo":["shampoo"],
    "snacks":["snacks"],
    "soap":["soap"],
    "sponge":["sponge"],
    "spoon":["spoon"],
    "tea":["tea"],
    "teaspoon":["teaspoon", "tea spoon", "tea spoons", "spoon"],
    "trash":["garbage", "debris", "trash"],
    "tray":["tray"],
    "toothpaste":["toothpaste"],
    "towel":["towel"],
    "tuna":["tuna", "tuna fish"],
    "water":["water"]
}

furniture_names_dict = {
    "bar":["bar"],
    "bed":["bed"],
    "bin":["bin"],
    "bookshelf":["bookshelf", "shelf", "bookcase"],
    "bowl":["bowl", "bowls"],
    "cab":["cab", "uber"],
    "cabinet":["cabinet"],
    "chair":["chair", "armchair", "baby chair"],
    "couch":["couch", "TV couch"],
    "counter":["counter"],
    "cupboard":["cupboard"],
    "desk":["desk"],
    "dishwasher":["dishwasher", "dish washer", "washer"],
    "dining_table":["dining table"],
    "drawer":["drawer", "cutlery drawer"],
    "dresser":["dresser"],
    "end_table":["end table"],
    "fireplace":["fireplace"],
    "fridge":["fridge", "freezer"],
    "microwave":["microwave"],
    "nightstand":["nightstand"],
    "shower":["shower"],
    "sink":["sink"],
    "sofa":["sofa"],
    "stove":["stove"],
    "side_table":["side table"],
    "storage_table":["storage table"],
    "shelf":["shelf"],
    "tall_table":["tall table"],
    "table":["table", "coffee table", "center table"], # "table",
    "towel_rack":["towel rack"],
    "tub":["tub", "bathtub"],
    "washing_machine":["washing machine", "wash"]
}

room_names_dict = {
    "bathroom":["bathroom"],
    "bedroom":["bedroom"],
    "dining_room":["dining room","dining"],
    "kitchen":["kitchen"],
    "living_room":["living room", "living"]
}

location_place_names_dict = {
    "bar":["bar"],
    "corridor":["corridor"], #"CORRIDOR",
    "entrance":["entrance", "exit"],
    "hallway":["hallway"],
    "toilet":["toilet"],
    "wardrobe":["wardrobe"],
    "uber":["uber"]
}

result_dict = {
    "task":"<none>",
    "target":"<none>",
    "prep_T1":"<none>",
    "location_T1":"<none>",
    "prep_T2":"<none>",
    "location_T2":"<none>",
    "room_T":"<none>",
    "destination":"<none>",
    "prep_D1":"<none>",
    "location_D1":"<none>",
    "prep_D2":"<none>",
    "location_D2":"<none>",
    "room_D":"<none>",
    "WYS":"<none>",
    "FIND":"<none>",
    "obj_option":"<none>",
    "obj_num":"<none>",
    "guesture":"<none>",
    "room_F":"<none>"
}

