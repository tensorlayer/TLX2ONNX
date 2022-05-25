#! /usr/bin/python
# -*- coding: utf-8 -*-

import time, sys

levels = {0: 'ERROR', 1: 'WARNING', 2: 'INFO', 3: 'DEBUG'}
class logging():
    log_level = 2

    @staticmethod
    def log(level=2, message="", use_color=False):
        current_time = time.time()
        time_array = time.localtime(current_time)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
        if logging.log_level >= level:
            if use_color:
                print("\033[1;31;40m{} [{}]\t{}\033[0m".format(
                    current_time, levels[level], message).encode("utf-8")
                      .decode("latin1"))
            else:
                print("{} [{}]\t{}".format(current_time, levels[level], message)
                      .encode("utf-8").decode("latin1"))
            sys.stdout.flush()

    @staticmethod
    def debug(message="", use_color=False):
        logging.log(level=3, message=message, use_color=use_color)

    @staticmethod
    def info(message="", use_color=False):
        logging.log(level=2, message=message, use_color=use_color)

    @staticmethod
    def warning(message="", use_color=True):
        logging.log(level=1, message=message, use_color=use_color)

    @staticmethod
    def error(message="", use_color=True, exit=True):
        logging.log(level=0, message=message, use_color=use_color)
        if exit:
            sys.exit(-1)