"""
config file for RSP project

date created: February 3rd 2021

author: Ziv Zango
"""
NOISES_DIR = "./noises/"
RECORDING_DIR = "./recordings/"
MAT_DIR = NOISES_DIR + "mat"
RESULTS_DIR = "results"
MARKERS = ['o', '^', 's', 'p', 'X']
COLORS = ['deepskyblue', 'mediumvioletred', 'mediumspringgreen', 'navy', 'orange']
NOISE_NAMES = []
NOISE_FS = 19980
INT2FLOAT_CONST = 32768.0
FS = 22050
SNR_LIST = [-10, -5, 0, 5, 10]
FILTER_SIZE_LIST = [16, 32, 64]
FILTER_SIZE = 32
FILTER_LEARNING_RATE_LIST = [0.01, 0.005, 0.001]
FILTER_LEARNING_RATE = 0.01
