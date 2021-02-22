"""
ssss
"""
import numpy
import librosa


class SpectralSubtraction:

    def __init__(self, noisy_signal, noise):
        self.__stft(noisy_signal, noise)
        self.exp_angle = numpy.exp(1.0j * numpy.angle(self.x))
        self.x = numpy.abs(self.x)
        self.d = numpy.mean(numpy.abs(self.d), axis=1)

    def __stft(self, s, n):
        self.x = librosa.stft(s)
        self.d = librosa.stft(n)

    def __istft(self):
        return librosa.istft(self.x)

    def calculate(self, alpha=4, beta=0.001):
        for i in range(self.x.shape[1]):
            for k in range(self.x.shape[0]):
                if numpy.square(self.x[k][i]) >= (alpha * self.d[k]):
                    self.x[k][i] = numpy.sqrt(numpy.square(self.x[k][i]) - (alpha * self.d[k]))
                else:
                    self.x[k][i] = numpy.sqrt(beta * self.d[k])
        self.x = self.x.astype(numpy.complex128) * self.exp_angle
        return self.__istft()
