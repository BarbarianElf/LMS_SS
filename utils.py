"""
utils for RSP project

date created: February 3rd 2021

author: Ziv Zango
"""

import numpy
import random
import adaptfilt
from matplotlib import pyplot as plt
from spectral_subtraction import SpectralSubtraction
from scipy.signal import coherence
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


if __name__ == '__main__':

    # get_mat_files()
    # for file in _get_wav_files(RECORDING_DIR):
    for noise_name in NOISE_NAMES:
        print(noise_name)
        for index, snr_db in enumerate(SNR_LIST):
            print(f"SNR dB: {snr_db}")
            re, n = get_signals("word.wav", snr_db, noise_name)
            # filter_size = 64
            filtered_signal = adaptive_noise_cancellation(n, re, 32, 0.01, normalized=False)
            # y, e, w = adaptfilt.nlms(n, de, filter_size, 0.1)
            # e = numpy.concatenate((e[0:filter_size-1], e))
            # e = numpy.pad(e, (filter_size - 1, 0))
            plt.figure(index)
            plt.plot(numpy.linspace(0, len(re) / FS, num=len(re)), re + n, label=f"SIGNAL+NOISE({snr_db}dB)")
            plt.plot(numpy.linspace(0, len(re) / FS, num=len(re)), filtered_signal, label="SIGNAL+NOISE FILTERED")
            plt.plot(numpy.linspace(0, len(re) / FS, num=len(re)), re, label="SIGNAL")
            plt.title(f"speech - noisy - filtered\nNOISE: {noise_name}")
            plt.xlabel("time (sec)")
            plt.legend()
            print(calc_snr_after(re, filtered_signal))
            new = SpectralSubtraction(filtered_signal, n).calculate()
            new = numpy.pad(new, (0, abs(len(re)-len(new))))
            plt.figure(index+1)
            plt.plot(numpy.linspace(0, len(re) / FS, num=len(re)), re, label="SIGNAL")
            plt.plot(numpy.linspace(0, len(re) / FS, num=len(re)), new, label=f"SIGNAL+NOISE({snr_db}dB)+LMS+SS")
            plt.title(f"speech - filtered & SS\nNOISE: {noise_name}")
            plt.xlabel("time (sec)")
            plt.legend()
            a, a1 = coherence(new, re, fs=FS)
            b, b1 = coherence(filtered_signal, re, fs=FS)
            plt.figure()
            plt.plot(a, a1, label='LMS-SS')
            plt.plot(b, b1, label='LMS')
            print("hi")
            plt.show()
            files_utils.save_data_to_wav(re + n, 'noisy')
            files_utils.save_data_to_wav(filtered_signal, 'afterLMS')
            files_utils.save_data_to_wav(new, 'afterLMSSS')
            print(calc_snr_after(re, new))

