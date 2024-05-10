import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

def fft_wav(input_file, output_file):
    # Read the WAV file
    sample_rate, data = wavfile.read(input_file)

    # Perform FFT
    fft_result = np.fft.fft(data)

    # Write the FFT result to a file
    with open(output_file, 'w') as f:
        f.write("Sample Rate: {}\n".format(sample_rate))
        f.write("FFT Result:\n")
        for value in fft_result:
            f.write("{}\n".format(value))

    # Visualize the spectrum
    spectrum = np.abs(fft_result)
    frequency_axis = np.fft.fftfreq(len(spectrum), d=1/sample_rate)

    plt.figure()
    plt.plot(frequency_axis, spectrum)
    plt.title('FFT Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

# Example usage
input_file = "convolved_output.wav"
output_file = "CONVFFT.txt"
#input_file = "convolved_output.wav"
#output_file = "correct_conv.txt"
fft_wav(input_file, output_file)
