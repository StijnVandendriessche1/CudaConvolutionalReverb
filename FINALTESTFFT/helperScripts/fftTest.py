import numpy as np
import scipy.io.wavfile as wavfile

def load_txt(filename):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, dtype=complex)
    return data

def load_txt_complex(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f:
            real, imag = map(float, line.split())
            data.append(complex(real, imag))
    return np.array(data)

def convolve_wav_from_txt(output_file, audio_fft_file, ir_fft_file, sample_rate):
    # Load FFT data from text files
    #fft_audio = load_txt(audio_fft_file)
    #fft_ir = load_txt(ir_fft_file)
    fft_audio = load_txt_complex(audio_fft_file)
    fft_ir = load_txt_complex(ir_fft_file)

    # Perform element-wise multiplication in the frequency domain
    convolved_fft = fft_audio * fft_ir

    # Perform IFFT to get the convolved signal
    convolved_signal = np.fft.ifft(convolved_fft).real

    # Normalize the convolved signal to prevent clipping
    max_val = np.max(np.abs(convolved_signal))
    if max_val > 0:
        convolved_signal /= max_val

    # Convert to 16-bit integer (PCM format)
    convolved_signal_int16 = (convolved_signal * 32767).astype(np.int16)

    # Write the convolved signal to a WAV file
    wavfile.write(output_file, sample_rate, convolved_signal_int16)


# Example usage
output_file = "convolved_output.wav"
audio_fft_file = "fftResult1.txt"
ir_fft_file = "fftResult2.txt"
sample_rate = 8000  # Sample rate assumed, adjust accordingly
convolve_wav_from_txt(output_file, audio_fft_file, ir_fft_file, sample_rate)
