#!/usr/bin/env python3
# coding:utf-8

from predict import CommandAnalyzer


def command_analyze(input_sentence):
    result =command_analyzer.predict(input_sentence)
    # for key, val in result.items():
    #     print(key, ":", val)
    return result


if __name__ == "__main__":
    command_analyzer = CommandAnalyzer()
    while True:
        try:
            input_str = input("please input command >>")
            result = command_analyze(input_str)
            for key, val in result.items():
                print(key, ":", val)
        except KeyboardInterrupt:
            break
