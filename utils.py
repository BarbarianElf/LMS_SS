import scipy.io
import librosa
import numpy
import soundfile
import os
import random
import adaptfilt
from matplotlib import pyplot as plt
from spectral_subtraction import SpectralSubtraction as SS


NOISES_DIR = "./noises/"
RECORDING_DIR = "./recordings/"
MAT_DIR = NOISES_DIR + "mat"
NOISE_NAMES = ["factory1"]
NOISE_FS = 19980
INT2FLOAT_CONST = 32768.0
FS = 22050
SNR_LIST = [-10, -5, 0, 5, 10]


def _get_wav_files(dir_path):
    """
    list all the files with wav extension in `dir_path`
    """
    files = []
    for file in os.listdir(dir_path):
        if file.endswith(".wav"):
            files.append(file)
    return files


def get_mat_files(dir_path=MAT_DIR, convert_to_wav=False):
    files = []
    for file in os.listdir(dir_path):
        if file.endswith(".mat"):
            noise_name = file.split('.')[0]
            NOISE_NAMES.append(noise_name)
            if convert_to_wav:
                convert_mat_to_wav(f"{dir_path}/{file}", noise_name)
            files.append(file)
    return files


def convert_mat_to_wav(mat_file, name):
    mat_data = scipy.io.loadmat(mat_file)
    wav_data = mat_data[name]
    if wav_data.dtype.name == 'int16':
        wav_data = wav_data.astype(numpy.float64, order='C') / INT2FLOAT_CONST
    soundfile.write(f"./noises/{name}.wav", wav_data, samplerate=NOISE_FS)


def get_wav_data(file, fs):
    data, _ = librosa.core.load(file, sr=fs)
    return data


def mean_square(x):
    return numpy.mean(numpy.square(x))


def calc_snr_factor(record_data, snr):
    return mean_square(record_data) / (pow(10, float(snr / 10.0)))


def get_noise_amp(noise_data, record_data, snr):
    start_index = random.randint(0, len(noise_data) - len(record_data))
    noise_data = noise_data[start_index: start_index + len(record_data)]
    snr_factor = calc_snr_factor(record_data, snr)
    noise_power = mean_square(noise_data)
    return noise_data * (snr_factor / noise_power)


def foo(file, snr):
    record_data = get_wav_data(RECORDING_DIR + file, FS)
    for noise in NOISE_NAMES:
        noise_data = get_wav_data(f"{NOISES_DIR}{noise}.wav", NOISE_FS)
        noise_amp = get_noise_amp(noise_data, record_data, snr)
        d = record_data + noise_amp
        return d, record_data, noise_amp


if __name__ == '__main__':
    # get_mat_files()
    # for file in _get_wav_files(RECORDING_DIR):
    for index, snr_db in enumerate(SNR_LIST):
        de, re, n = foo("Hello.wav", snr_db)
        filter_size = 64
        y, e, w = adaptfilt.nlms(n, de, filter_size, 0.1)
        e = numpy.pad(e, (filter_size - 1, 0))
        plt.figure(index)
        plt.plot(numpy.linspace(0, len(de) / FS, num=len(de)), de)
        # plt.plot(numpy.linspace(0, len(de) / FS, num=len(de)), re)
        # plt.figure(index + 1)
        # plt.plot(y)
        plt.plot(numpy.linspace(0, len(de) / FS, num=len(de)), e)
        plt.plot(numpy.linspace(0, len(de) / FS, num=len(de)), re)

        new = SS(e, n).calculate()
        plt.figure(index+1)
        new = numpy.pad(new, (len(re)-len(new), 0))
        plt.plot(numpy.linspace(0, len(de) / FS, num=len(de)), re)
        plt.plot(numpy.linspace(0, len(de) / FS, num=len(de)), new)

        soundfile.write("newGerman_Sheperdf16.wav", new, FS)
        plt.show()
        print("hi")
