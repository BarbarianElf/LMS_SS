import numpy
import files_utils
import utils
from config import *


def get_noises_names(dir_path):
    for noise_file in files_utils.get_files(dir_path, "wav"):
        noise_name = noise_file.split('.')[0]
        NOISE_NAMES.append(noise_name)
    return True


def optimized_lms_filter(noise_name, snr_db):
    best_snr = -20
    audio_file = "word.wav"
    learning_rate_opt, filter_size_opt = FILTER_LEARNING_RATE_LIST[0], FILTER_SIZE_LIST[0]
    for lr in FILTER_LEARNING_RATE_LIST:
        for f_size in FILTER_SIZE_LIST:
            real_speech, noise = utils.get_signals(audio_file, snr_db, noise_name)
            filtered_signal = utils.adaptive_noise_cancellation(noise, real_speech, f_size, lr, normalized=False)
            snr_after = utils.calc_snr_after(real_speech, filtered_signal)
            if snr_after > best_snr:
                best_snr = snr_after
                learning_rate_opt, filter_size_opt = lr, f_size
    return learning_rate_opt, filter_size_opt


if __name__ == '__main__':
    # Only for the first time: Coverts mat file to wav
    # for file in files_utils.get_files(MAT_DIR, "mat"):
    #     files_utils.convert_mat_to_wav(MAT_DIR + file, NOISES_DIR, fs=NOISE_FS)

    # Get Noises names from noises wav files directory into NOISE_NAMES list
    get_noises_names(NOISES_DIR)

    # Get the optimal LEARNING RATE and FILTER SIZE for LMS
    lr_list, f_size_list = [], []
    for snr in SNR_LIST:
        for noisy in NOISE_NAMES:
            learning_rate, filter_size = optimized_lms_filter(noisy, snr)
            lr_list.append(learning_rate)
            f_size_list.append(filter_size)
    learning_rate = numpy.mean(lr_list)
    filter_size = numpy.mean(f_size_list)

    print("hi")
