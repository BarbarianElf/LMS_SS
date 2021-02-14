import os
import scipy.io
import numpy
import soundfile

INT2FLOAT_CONST = 32768.0


def get_mat_files(dir_path, fs, convert_to_wav=False):
    files = []
    for file in os.listdir(dir_path):
        if file.endswith(".mat"):
            noise_name = file.split('.')[0]
            if convert_to_wav:
                _convert_mat_to_wav(f"{dir_path}/{file}", noise_name, fs)
            files.append(file)
    return files


def _convert_mat_to_wav(mat_file, name, fs):
    mat_data = scipy.io.loadmat(mat_file)
    wav_data = mat_data[name]
    if wav_data.dtype.name == 'int16':
        wav_data = wav_data.astype(numpy.float64, order='C') / INT2FLOAT_CONST
    soundfile.write(f"./noises/{name}.wav", wav_data, samplerate=fs)
