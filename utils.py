"""
utils for RSP project

date created: February 3rd 2021

author: Ziv Zango
"""

import numpy
import random
import adaptfilt
from matplotlib import pyplot as plt

import files_utils
from config import *


def get_plt():
    return plt


def calc_alpha(snr, alpha):
    if (snr >= -5) and (snr <= 20):
        return alpha - (3 * snr / 20)
    return alpha


def mean_square(x):
    return numpy.mean(numpy.square(x))


def calc_snr_factor(record_data, snr):
    return mean_square(record_data) / (pow(10, float(snr / 10.0)))


def calc_snr_after(signal, noise_reduced_signal, db=True):
    snr_after = mean_square(signal) / mean_square(noise_reduced_signal - signal)
    if db:
        return 10 * numpy.log10(snr_after)
    return snr_after


def get_noise_amp(noise_data, record_data, snr):
    start_index = random.randint(0, len(noise_data) - len(record_data))
    noise_data = noise_data[start_index: start_index + len(record_data)]
    snr_factor = calc_snr_factor(record_data, snr)
    noise_power = mean_square(noise_data)
    return noise_data * (snr_factor / noise_power)


def adaptive_noise_cancellation(noise, signal, filter_size=64, learning_rate=0.1, normalized=True):
    if normalized:
        y, e, w = adaptfilt.nlms(noise, noise + signal, filter_size, learning_rate)
    else:
        y, e, w = adaptfilt.lms(noise, noise + signal, filter_size, learning_rate)
    return numpy.concatenate((e[0:filter_size - 1], e))


def get_signals(file, snr, noise):
    record_data = files_utils.get_wav_data(RECORDING_DIR + file, FS)
    noise_data = files_utils.get_wav_data(f"{NOISES_DIR}{noise}.wav", NOISE_FS)
    noise_amp = get_noise_amp(noise_data, record_data, snr)
    return record_data, noise_amp
