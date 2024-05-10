import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def load_txt(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f:
            real, imag = map(float, line.split())
            data.append(complex(real, imag))
    return np.array(data)

def plot_spectrum(filename, title):
    fft_data = load_txt(filename)
    spectrum = np.abs(fft_data)
    sample_rate = 8000
    num_samples = len(spectrum)
    frequency_axis = np.fft.fftfreq(num_samples, d=1/sample_rate)

    plt.figure()
    plt.plot(frequency_axis, spectrum)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

def calculate_fft(audio_file):
    rate, data = wavfile.read(audio_file)
    fft_data = np.fft.fft(data)
    return fft_data

def visualize_fft_spectrum(fft_data, title):
    spectrum = np.abs(fft_data)
    sample_rate = 8000
    num_samples = len(spectrum)
    frequency_axis = np.fft.fftfreq(num_samples, d=1 / sample_rate)

    plt.figure()
    plt.plot(frequency_axis, spectrum)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

def multiply_spectrums(spectrum1, spectrum2):
    return spectrum1 * spectrum2

# Read txt spectra
ir_spectrum = load_txt('fftResult2.txt')
audio_spectrum = load_txt('fftResult1.txt')

#visualize txt spectra
visualize_fft_spectrum(ir_spectrum, "IR TXT SPECTRUM")
visualize_fft_spectrum(audio_spectrum, "AUDIO TXT SPECTRUM")

# Read the audio and impulse response WAV files
sample_rate_audio, data_audio = wavfile.read('AUDIO2.wav')
sample_rate_ir, data_ir = wavfile.read('IR.wav')

# Determine the length of the resulting signal after convolution
output_length = len(data_audio) + len(data_ir) - 1

# Perform zero-padding to make both signals compatible for convolution
pad_length_audio = output_length - len(data_audio)
pad_length_ir = output_length - len(data_ir)

padded_audio = np.pad(data_audio, (0, pad_length_audio), mode='constant')
padded_ir = np.pad(data_ir, (0, pad_length_ir), mode='constant')

# Perform FFT on both signals
fft_audio = np.fft.fft(padded_audio)
fft_ir = np.fft.fft(padded_ir)

# Visualize product spectra
visualize_fft_spectrum(fft_audio, 'AUDIO WAV SPECTRUM')
visualize_fft_spectrum(fft_ir, 'IR WAV SPECTRUM')

# Multiply spectra
txt_spectrum_product = multiply_spectrums(ir_spectrum, audio_spectrum)
wav_spectrum_product = multiply_spectrums(fft_ir, fft_audio)

# Visualize product spectra
visualize_fft_spectrum(txt_spectrum_product, 'TXT Spectrum Product')
visualize_fft_spectrum(wav_spectrum_product, 'Wav Spectrum Product')
