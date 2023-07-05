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
    "tidyup":1,
    "bring":2,
    "find_object":3,
    "find_person":4,
    "follow":5,
    "guide":6,
    "manipulate":7,
    "introduce":8,
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
    " supplyes ":" supply "
}

replace_phrases = {
    "what's":"what is",
    "who's":"who is",
    "he's":"he is",
    "there's":"there is",
    "team's":"team is",
    "i'll":"i will",
    "i'd":"i would",
    "choco flake":"chocoflake",
    "choco flakes":"chocoflake",
    "choco cereal":"chocoflake",
    "choco cereals":"chocoflake",
    "tea spoon":"teaspoon",
    "bookcase":"bookshelf",
    "book case":"bookshelf",
    "book shelf":"bookshelf",
    "could you ":"",
    "cheese it":"cheezit"
    # "chocolate drink":"chocodrink",
    # "cleaning supply":"cleaning_supply",
    # "cleaning supplies":"cleaning_supply",
    # "cleaning stuff":"cleaning_supply",
    # "cleaning stuffes":"cleaning_supply",
    # "dining table":"diningtable",
    # "bubble tea":"bubbletea",
    # "ice tea":"icetea",
    # "couch table":"couchtable",#couchtable
    # "TV couch":"couch",
    # "kitchen shelf":"kitchenshelf",#kitchenshelf
    # "small shelf":"smallshelf",#smallshelf
    # "big shelf":"bigshelf",#"bigshelf"
    # "dish washer":"dishwasher",#"dishwasher"
    # "office shelf":"officeshelf"#"officeshelf"
}


gender_dict = {
    "male":["male", "males", "boy", "boys", "man", "men", "gentleman", "gentlemen",],
    "female":["female", "females", "girl", "girls", "woman", "women", "lady", "ladies"]
    }

item_names_dict = {
    "apple":["apple"], #fruits
    "banana":["banana"],
    "fruit":["fruit", "fruits"],
    "lemon":["lemon"],
    "orange":["orange"],
    "peach":["peach"],
    "pear":["pear"],
    "plum":["plum"],
    "strawberry":["strawberry"],
    "cleaning_supply":["cleaning stuff","cleaning supply","cleaning supplies"], #clean
    "cleanser":["cleanser"],
    "sponge":["sponge"],
    "bowl":["bowl", "bowls"], #dishes
    "cup":["cup", "cups"],
    "dish":["dish", "dishes"],
    "fork":["fork"],
    "knife":["knife"],
    "plate":["plate"],
    "spoon":["spoon", "tea spoon"],
    # "teaspoon":["tea spoon", "tea spoons", "spoon"],
    "cola":["coke", "cokes", "cola"], #drinks
    "iced_tea":["iced tea"],
    "juice_pack":["juice", "juice pack"],
    "milk":["milk"],
    "orange_juice":["orange juice"],
    "red_wine":["red wine", "wine"],
    "tropical_juice":["tropical juice"],
    "chocolate_jello":["chocolate jello"], #food
    "coffee_grounds":["coffee grounds"],
    "mustard":["mustard"],
    "spam":["spam"],
    "strawberry_jello":["strawberry jello"],
    "sugar":["sugar"],
    "tomato_soap":["tomato soap", "soap"],
    "tuna":["tuna", "tuna can"],
    "snacks":["snacks"], #snacks
    "cheezit":["cheezit"],
    "cornflakes":["cereal", "flakes", "corn flakes"],
    "pringles":["pringles", "chips", "potato chips"],
    "baseball":["baseball"],
    "dice":["dice", "dices"],
    "rubiks_cube":["rubiks cube"],
    "soccer_ball":["soccer ball"],
    "tennis_ball":["tennis ball"],
    "bag":["bag"],
    # "crackers":["crackers"],
    # "potato_chips":["potato chips"],
    # "paprika":["paprika"],
    # "noodle":["noodle"],#"noodles", 
    # "sausages":["sausages"],
    # "beer":["beer", "beers"],
    "box":["box"],
    # "bubble_tea":["bubble tea"],
    # "block":["block", "blocks"],
    # "chips":["chips", "potato chips"],
    # "chocoflake":["choco cereal", "choco flakes"], #"chocoflake", "chocoflakes", "flakes of choco", 
    # "chocolate":["chocolate"],
    # "chocodrink":["chocolate drink"],
    # "cloth":["cloth"],
    # "cookies":["cookie", "cookies"],
    # "coffee_jar":["coffee jar"],
    "cutlery":["cutlery"],
    # "detergent":["detergent", "detergents"],
    "drink":["drink", "drinks"],
    # "glass":["glass"],
    "key":["key"],
    # "lunch_box":["lunch box", "lunch boxes"],
    # "light_bulb":["light bulb", "light bulbs"],
    # "melon":["melon"],
    # "mug":["mug"],
    # "napkin":["napkin"],
    # "pasta":["pasta", "spaghetti"],
    "pantry_items":["pantry items"],
    # "pickle":["pickles", "pickle"],
    # "pocky":["pockys", "pocky"],
    # "shampoo":["shampoo"],
    # "soap":["soap"],
    # "tea":["tea"],
    "trash":["debris", "trash"],
    "tray":["tray"],
    # "toothpaste":["toothpaste"],
    # "towel":["towel"],
    # "tonic":["tonic"],
    # "water":["water"],
    "object":["object", "objects", "item", "items"]
}

furniture_names_dict = {
    "bar":["bar"],
    "bed":["bed"],
    "bin":["bin"],
    "bookshelf":["bookshelf", "bookcase"],
    "big_shelf":["big shelf"],
    "cab":["cab", "uber"],
    "cabinet":["cabinet"],
    "chair":["chair", "armchair"],
    "couch_table":["couch table"],
    "couch":["couch", "TV couch"],
    "counter":["counter"],
    "cupboard":["cupboard"],
    "coat_rack":["coat rack"],
    "desk":["desk"],
    "dishwasher":["dishwasher", "washer"],
    "dining_table":["dining table"],
    # "dinner_table":["dinner table"],
    "drawer":["drawer", "cutlery drawer"],
    "dresser":["dresser"],
    "end_table":["end table"],
    # "fireplace":["fireplace"],
    "fridge":["fridge", "freezer"],
    "house_plant":["house plant"],
    "kitchen_shelf":["kitchen shelf"],
    "kitchen_bin":["kitchen bin"],
    "microwave":["microwave"],
    "nightstand":["nightstand"],
    "pantry":["pantry"],
    "show_rack":["show rack"],
    "shower":["shower"],
    "sink":["sink"],
    "sofa":["sofa"],
    "stove":["stove"],
    "side_table":["side table"],
    "storage_table":["storage table"],
    "small_shelf":["small shelf"],
    "shelf":["shelf"],
    # "office_shelf":["office shelf"],
    "tall_table":["tall table"],
    "table":["table", "center table"], # "table",, "coffee table"
    # "towel_rack":["towel rack"],
    "night_table":["night table"],
    "tub":["tub", "bathtub"],
    "tv":["tv"],
    "washing_machine":["washing machine"]#, "wash"
}

room_names_dict = {
    # "bathroom":["bathroom"],
    "bedroom":["bedroom"],
    "study_room":["study room"],
    "kitchen":["kitchen"],
    "living_room":["living room", "living"]
    # "office":["office"]
}

location_place_names_dict = {
    "bar":["bar"],
    "corridor":["corridor"], #"CORRIDOR",
    "entrance":["entrance"],
    "exit":["exit"],
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
    "gesture":"<none>",
    "room_F":"<none>"
}

