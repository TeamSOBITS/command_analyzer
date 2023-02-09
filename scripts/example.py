#!/usr/bin/env python3
# coding:utf-8
from predict import CommandAnalyzer


if __name__ == "__main__":
    command_analyzer = CommandAnalyzer()
    while True:
        try:
            input_str = input("please input command >>")
            result =command_analyzer.predict(input_str)
            for key, val in result.items():
                print(key, ":", val)
        except KeyboardInterrupt:
            break
