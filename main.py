import numpy
from spectral_subtraction import SpectralSubtraction
import files_utils
import utils
from config import *


def get_noises_names(dir_path):
    for noise_file in files_utils.get_files(dir_path, "wav"):
        noise_name = noise_file.split('.')[0]
        NOISE_NAMES.append(noise_name)
    return True


def optimized_lms_filter(noise_name, snr_db, best_snr=-20, audio_file="word.wav"):
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


def optimized_ss_after_lms(noise_name, snr_db, alpha=4, audio_file="small_sentence.wav"):
    real_speech, noise = utils.get_signals(audio_file, snr_db, noise_name)
    filtered_signal = utils.adaptive_noise_cancellation(noise, real_speech, 32, 0.01, normalized=False)
    snr_after = utils.calc_snr_after(real_speech, filtered_signal)
    if (snr_after >= -5) and (snr_after <= 20):
        alpha = alpha - (3 * snr_after / 20)
    ss_filtered_signal_depended = SpectralSubtraction(filtered_signal, noise).calculate(alpha=alpha)
    ss_filtered_signal_constant = SpectralSubtraction(filtered_signal, noise).calculate(alpha=4)
    snr_after_depended = utils.calc_snr_after(real_speech, ss_filtered_signal_depended)
    snr_after_constant = utils.calc_snr_after(real_speech, ss_filtered_signal_constant)
    # print(f"snr depended: {snr_after_depended}\t snr constant: {snr_after_constant}")
    if snr_after_constant > snr_after_depended:
        return 1
    return 0


def lms_or_nlms(noise_name, snr_db, audio_file="medium_sentence.wav"):
    real_speech, noise = utils.get_signals(audio_file, snr_db, noise_name)
    filtered_signal_lms = utils.adaptive_noise_cancellation(noise, real_speech, 32, 0.01, normalized=False)
    filtered_signal_nlms = utils.adaptive_noise_cancellation(noise, real_speech, 32, 0.1, normalized=True)
    snr_lms = utils.calc_snr_after(real_speech, filtered_signal_lms)
    snr_nlms = utils.calc_snr_after(real_speech, filtered_signal_nlms)
    # print(f"snr lms: {snr_lms}\t snr nlms: {snr_nlms}")
    if snr_lms > snr_nlms:
        return 1
    return 0


if __name__ == '__main__':
    # Only for the first time: Coverts mat file to wav
    # for file in files_utils.get_files(MAT_DIR, "mat"):
    #     files_utils.convert_mat_to_wav(MAT_DIR + file, NOISES_DIR, fs=NOISE_FS)

    # Get Noises names from noises wav files directory into NOISE_NAMES list
    get_noises_names(NOISES_DIR)

    # Get the optimal LEARNING RATE and FILTER SIZE for LMS
    # lr_list, f_size_list = [], []
    # for snr in SNR_LIST:
    #     for noisy in NOISE_NAMES:
    #         learning_rate, filter_size = optimized_lms_filter(noisy, snr)
    #         lr_list.append(learning_rate)
    #         f_size_list.append(filter_size)
    # learning_rate = numpy.mean(lr_list)
    # filter_size = numpy.mean(f_size_list)
    # print(f"Average learning rate: {learning_rate}\nAverage filter size: {filter_size}")

    # Get the optimal alpha for Spectral Subtraction
    # alpha_constant, alpha_depended_snr = 0, 0
    # for snr in SNR_LIST:
    #     for noisy in NOISE_NAMES:
    #         if optimized_ss_after_lms(noisy, snr):
    #             alpha_constant += 1
    #         else:
    #             alpha_depended_snr += 1
    # if alpha_constant > alpha_depended_snr:
    #     print("Alpha constant (4) preferred")
    # else:
    #     print("Alpha depended SNR preferred")

    # Get which better LMS or NLMS
    # lms, nlms = 0, 0
    # for snr in SNR_LIST:
    #     for noisy in NOISE_NAMES:
    #         if lms_or_nlms(noisy, snr):
    #             lms += 1
    #         else:
    #             nlms += 1
    # if lms > nlms:
    #     print("LMS Filter preferred")
    # else:
    #     print("NLMS Filter preferred")
    print("hi")
