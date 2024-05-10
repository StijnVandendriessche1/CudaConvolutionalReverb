import numpy as np
import scipy.io.wavfile as wavfile


def convolve_wav(audio_file, ir_file, output_file):
    # Read the audio and impulse response WAV files
    sample_rate_audio, data_audio = wavfile.read(audio_file)
    sample_rate_ir, data_ir = wavfile.read(ir_file)

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
    wavfile.write(output_file, sample_rate_audio, convolved_signal_int16)


# Example usage
audio_file = "AUDIO2.wav"
ir_file = "IR.wav"
output_file = "convolved_output.wav"
convolve_wav(audio_file, ir_file, output_file)
