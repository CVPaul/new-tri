#!/usr/bin/env python
#encoding=utf-8
TOTAL_INDEX_CNT = 1000000
FIX_HISTORY_LENGTH = 7
START_HOLD_RATIO = 0.3

BASIC_POINT = 1996.26
BASIC_PRICE = 100.0

ITER_BATCH_SIZE = 100000
DUMP_EVERY_ITER = 100000
MAX_ITER_CNT = 10000000

ACTION_CNT = 20
INIT_VALUE = 0.0
DRAWDOWN_FAC = 2.0

data_prefix = "../data/"
path = "%sshangzheng-CSI000016.csv"%data_prefix

# switches
VERBOSE = True
DEBUG = False
PHASE = "Test"  # Train、Test、Both
REWARD_TYPE = "DrawDown" # DrawDown、Total