"""
files utils for RSP project

date created: February 3rd 2021

author: Ziv Zango
"""

import os
import scipy.io
import numpy
import soundfile
import librosa

from config import INT2FLOAT_CONST


def get_files(dir_path, extension):
    """
    list all the files with wav extension in `dir_path`
    """
    files = []
    for file in os.listdir(dir_path):
        if file.endswith(f".{extension}"):
            files.append(file)
    return files


def get_wav_data(file, fs):
    data, _ = librosa.core.load(file, sr=fs)
    return data


def convert_mat_to_wav(file, path_destination, fs):
    if not file.endswith(".mat"):
        raise NameError("File needs to be a .mat file")
    name = file.split('.')[0]
    mat_data = scipy.io.loadmat(file)
    wav_data = mat_data[name]
    if wav_data.dtype.name == 'int16':
        wav_data = wav_data.astype(numpy.float64, order='C') / INT2FLOAT_CONST
    soundfile.write(f"{path_destination}{name}.wav", wav_data, samplerate=fs)


def save_data_to_wav(data, name, fs=22050, path=os.path.dirname(os.path.abspath(__file__))):
    soundfile.write(f"{path}\\{name}.wav", data, fs)


def handle_folder(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    return True
