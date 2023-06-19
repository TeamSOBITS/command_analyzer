#!/usr/bin/env python3
# coding:utf-8

import rospy
from predict import CommandAnalyzer

def analyze_sentence(sentence):
    command_analyzer = CommandAnalyzer()
    while True:
        try:
            input_str = input("please input command >>")
            result =command_analyzer.predict(input_str)
            for key, val in result.items():
                print(key, ":", val)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    rospy.init_node('command_analyzer_node') 
    rospy.loginfo('Hello World')
    example = "robot please meet Alex at the desk and follow her to the kitchen"
    analyze_sentence(example)
