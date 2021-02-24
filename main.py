import numpy
from scipy.signal import coherence

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
    filtered_signal = utils.adaptive_noise_cancellation(noise,
                                                        real_speech,
                                                        FILTER_SIZE,
                                                        FILTER_LEARNING_RATE,
                                                        normalized=False)
    snr_after = utils.calc_snr_after(real_speech, filtered_signal)
    alpha = utils.calc_alpha(snr_after, alpha)
    ss_filtered_signal_depended = SpectralSubtraction(filtered_signal, noise).calculate(alpha=alpha)
    ss_filtered_signal_constant = SpectralSubtraction(filtered_signal, noise).calculate(alpha=4)
    ss_filtered_signal_depended = numpy.pad(ss_filtered_signal_depended,
                                            (0, abs(len(real_speech)-len(ss_filtered_signal_depended))))
    ss_filtered_signal_constant = numpy.pad(ss_filtered_signal_constant,
                                            (0, abs(len(real_speech)-len(ss_filtered_signal_constant))))
    snr_after_depended = utils.calc_snr_after(real_speech, ss_filtered_signal_depended)
    snr_after_constant = utils.calc_snr_after(real_speech, ss_filtered_signal_constant)
    # print(f"snr depended: {snr_after_depended}\t snr constant: {snr_after_constant}")
    if snr_after_constant > snr_after_depended:
        return 1
    return 0


def lms_or_nlms(noise_name, snr_db, audio_file="medium_sentence.wav"):
    real_speech, noise = utils.get_signals(audio_file, snr_db, noise_name)
    filtered_signal_lms = utils.adaptive_noise_cancellation(noise, real_speech, FILTER_SIZE, 0.01, normalized=False)
    filtered_signal_nlms = utils.adaptive_noise_cancellation(noise, real_speech, FILTER_SIZE, 0.1, normalized=True)
    snr_lms = utils.calc_snr_after(real_speech, filtered_signal_lms)
    snr_nlms = utils.calc_snr_after(real_speech, filtered_signal_nlms)
    # print(f"snr lms: {snr_lms}\t snr nlms: {snr_nlms}")
    if snr_lms > snr_nlms:
        return 1
    return 0


def coherence_example(x, n, y1, y2):
    f, c_init = coherence(x, x + n, fs=FS, nfft=256*4)
    _, c_lms = coherence(x, y1, fs=FS, nfft=256*4)
    _, c_lms_ss = coherence(x, y2, fs=FS, nfft=256*4)
    plt.figure(20)
    plt.title("Coherence example")
    plt.xlabel("frequency [Hz]")
    plt.ylabel("Value")
    plt.plot(f, c_init[1:48 * 4], label=r'$C_{xy}(\frac{\omega}{2\pi})$')
    plt.plot(f, c_lms[1:48 * 4], label=r'$C_{xy}(\frac{\omega}{2\pi}) - LMS$')
    plt.plot(f, c_lms_ss[1:48 * 4], label=r'$C_{xy}(\frac{\omega}{2\pi}) - LMSSS$')
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    plt = utils.get_plt()
    files_utils.handle_folder(RESULTS_DIR)
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
    k = 0
    for index, record in enumerate(files_utils.get_files(RECORDING_DIR, "wav")):
        name = record.split('.')[0]
        plt.figure(index + k)
        plt.title(f"SNR in-out {name}\nLMS")
        plt.xlabel("SNR[dB] in")
        plt.ylabel("SNR[dB] out")
        plt.figure(index + k + 1)
        plt.title(f"SNR in-out {name}\nLMS-SS")
        plt.xlabel("SNR[dB] in")
        plt.ylabel("SNR[dB] out")
        plt.figure(index + k + 2)
        plt.title(f"Coherence {name}\nLMS")
        plt.xlabel("SNR[dB] in")
        plt.ylabel("Value")
        plt.figure(index + k + 3)
        plt.title(f"Coherence {name}\nLMS-SS")
        plt.xlabel("SNR[dB] in")
        plt.ylabel("Value")
        for marker, noisy in enumerate(NOISE_NAMES):
            snr_lms_list, snr_lms_ss_list = [], []
            c_lms_list, c_lms_ss_list = [], []
            for snr in SNR_LIST:
                record_data, noise_data = utils.get_signals(record, snr, noisy)
                filtered_lms_signal = utils.adaptive_noise_cancellation(noise_data,
                                                                        record_data,
                                                                        FILTER_SIZE,
                                                                        FILTER_LEARNING_RATE,
                                                                        normalized=False)
                snr_after_lms = utils.calc_snr_after(record_data, filtered_lms_signal)
                _, c_lms = coherence(record_data, filtered_lms_signal)
                snr_lms_list.append(snr_after_lms)
                c_lms_list.append(numpy.mean(c_lms[1:48 * 4]))
                a = utils.calc_alpha(snr_after_lms, alpha=4)
                lms_ss_signal_ = SpectralSubtraction(filtered_lms_signal, noise_data).calculate(alpha=a)
                lms_ss_signal = numpy.pad(lms_ss_signal_, (0, abs(len(record_data)-len(lms_ss_signal_))))
                # coherence_example(record_data, noise_data, filtered_lms_signal, lms_ss_signal)
                snr_after_lms_ss = utils.calc_snr_after(record_data, lms_ss_signal)
                _, c_lms_ss = coherence(record_data, lms_ss_signal)
                snr_lms_ss_list.append(snr_after_lms_ss)
                c_lms_ss_list.append(numpy.mean(c_lms_ss[1:48 * 4]))
            plt.figure(index + k)
            plt.plot(SNR_LIST, snr_lms_list,
                     color=COLORS[marker],
                     marker=MARKERS[marker],
                     linewidth=0.5,
                     label=f"{noisy}")
            plt.figure(index + k + 1)
            plt.plot(SNR_LIST, snr_lms_ss_list,
                     color=COLORS[marker],
                     marker=MARKERS[marker],
                     linewidth=0.5,
                     label=f"{noisy}")
            plt.figure(index + k + 2)
            plt.plot(SNR_LIST, c_lms_list,
                     color=COLORS[marker],
                     marker=MARKERS[marker],
                     linewidth=0.5,
                     label=f"{noisy}")
            plt.figure(index + k + 3)
            plt.plot(SNR_LIST, c_lms_ss_list,
                     color=COLORS[marker],
                     marker=MARKERS[marker],
                     linewidth=0.5,
                     label=f"{noisy}")
            print(name)
            print(f"snr_lms: {noisy}: {snr_lms_list}")
            print(f"snr_lms_ss: {noisy}: {snr_lms_ss_list}")
            print("\n")
        plt.figure(index + k)
        plt.grid(color='gainsboro')
        plt.legend()
        plt.ylim([-5, 42])
        plt.savefig(f"{RESULTS_DIR}/{name}_LMS_snr")
        plt.figure(index + k + 1)
        plt.grid(color='gainsboro')
        plt.legend()
        plt.ylim([-5, 42])
        plt.savefig(f"{RESULTS_DIR}/{name}_LMS-SS_snr")
        plt.figure(index + k + 2)
        plt.grid(color='gainsboro')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.savefig(f"{RESULTS_DIR}/{name}_LMS_coherence")
        plt.figure(index + k + 3)
        plt.grid(color='gainsboro')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.savefig(f"{RESULTS_DIR}/{name}_LMS-SS_coherence")
        k += 3
