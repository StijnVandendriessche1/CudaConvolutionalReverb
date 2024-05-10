#include <iostream>
#include <fstream>
#include <Windows.h>
#include <mmsystem.h>
#include <vector>
#include <complex>
#include <fftw3.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <string>

#pragma comment(lib, "Winmm.lib")
using namespace std;

bool handleAudioFiles(const char* inputFile1, const char* inputFile2, vector<short>& audioData1, vector<short>& audioData2) {
    MMIOINFO mmioInfo1, mmioInfo2;
    HMMIO hFile1, hFile2;

    // Open input files
    hFile1 = mmioOpenA(const_cast<char*>(inputFile1), NULL, MMIO_READ);
    hFile2 = mmioOpenA(const_cast<char*>(inputFile2), NULL, MMIO_READ);

    if (hFile1 == NULL || hFile2 == NULL) {
        cerr << "Error: Unable to open input files.\n";
        return false;
    }

    // Skip over the header information
    MMCKINFO chunkInfo1, chunkInfo2;
    if (mmioDescend(hFile1, &chunkInfo1, NULL, 0) != MMSYSERR_NOERROR ||
        mmioDescend(hFile2, &chunkInfo2, NULL, 0) != MMSYSERR_NOERROR) {
        cerr << "Error: Unable to read file info.\n";
        mmioClose(hFile1, 0);
        mmioClose(hFile2, 0);
        return false;
    }

    // Find the 'data' chunk
    while (chunkInfo1.ckid != mmioFOURCC('d', 'a', 't', 'a') &&
        chunkInfo2.ckid != mmioFOURCC('d', 'a', 't', 'a')) {
        if (mmioDescend(hFile1, &chunkInfo1, NULL, 0) != MMSYSERR_NOERROR ||
            mmioDescend(hFile2, &chunkInfo2, NULL, 0) != MMSYSERR_NOERROR) {
            cerr << "Error: Unable to find 'data' chunk.\n";
            mmioClose(hFile1, 0);
            mmioClose(hFile2, 0);
            return false;
        }
    }

    // Allocate memory for audio data
    audioData1.resize(chunkInfo1.cksize / sizeof(short));
    audioData2.resize(chunkInfo2.cksize / sizeof(short));

    // Read audio data from input files
    mmioRead(hFile1, reinterpret_cast<HPSTR>(audioData1.data()), chunkInfo1.cksize);
    mmioRead(hFile2, reinterpret_cast<HPSTR>(audioData2.data()), chunkInfo2.cksize);

    // Close input files
    mmioClose(hFile1, 0);
    mmioClose(hFile2, 0);

    // Display the number of samples read from each file
    cout << "Samples in file 1: " << audioData1.size() << endl;
    cout << "Samples in file 2: " << audioData2.size() << endl;

    return true;
}

// Function to pad a signal with zeros to a desired length
vector<short> zeroPadSignal(const vector<short>& signal, size_t desiredLength) {
    vector<short> paddedSignal(desiredLength, 0);
    size_t originalLength = signal.size();
    if (originalLength >= desiredLength) {
        // If the original signal is already longer than or equal to the desired length, return it unchanged
        return signal;
    }
    // Copy the original signal to the padded signal
    for (size_t i = 0; i < originalLength; ++i) {
        paddedSignal[i] = signal[i];
    }
    return paddedSignal;
}

// Function to perform FFT on a signal
vector<complex<double>> computeFFT(const vector<short>& signal) {
    // Prepare FFT input/output arrays
    int N = signal.size();
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

    // Copy signal data to FFT input array
    for (int i = 0; i < N; ++i) {
        in[i][0] = signal[i];
        in[i][1] = 0.0;
    }

    // Create FFT plan
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Execute FFT
    fftw_execute(plan);

    // Convert FFT output to vector<complex<double>>
    vector<complex<double>> fftResult(N);
    for (int i = 0; i < N; ++i) {
        fftResult[i] = complex<double>(out[i][0], out[i][1]);
    }

    // Clean up
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return fftResult;
}

// Function to perform element-wise multiplication of two vectors of complex numbers
vector<complex<double>> elementwiseMultiply(const vector<complex<double>>& vec1, const vector<complex<double>>& vec2) {
    // Check if the input vectors have the same size
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Input vectors must have the same size for element-wise multiplication.");
    }

    // Create a vector to store the result
    vector<complex<double>> result(vec1.size());

    // Perform element-wise multiplication
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] * vec2[i];
    }

    return result;
}

// Function to perform IFFT on a vector of complex numbers
vector<double> computeIFFT(const vector<complex<double>>& fftResult) {
    // Prepare IFFT input/output arrays
    int N = fftResult.size();
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

    // Copy FFT result data to IFFT input array
    for (int i = 0; i < N; ++i) {
        in[i][0] = fftResult[i].real();
        in[i][1] = fftResult[i].imag();
    }

    // Create IFFT plan
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Execute IFFT
    fftw_execute(plan);

    // Convert IFFT output to vector<double> and apply normalization
    vector<double> ifftResult(N);
    double normalizationFactor = 1.0 / N;
    for (int i = 0; i < N; ++i) {
        ifftResult[i] = out[i][0] * normalizationFactor; // Real part only
    }

    // Clean up
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return ifftResult;
}

// Function to normalize a vector of doubles
void normalizeVector(vector<double>& data) {
    double max_val = 0.0;
    // Find the maximum absolute value in the vector
    for (const auto& val : data) {
        max_val = max(max_val, abs(val));
    }
    // If max_val is greater than 0, normalize the vector
    if (max_val > 0.0) {
        for (auto& val : data) {
            val /= max_val;
        }
    }
}

// Function to convert a vector of doubles to 16-bit PCM format
vector<int16_t> convertToPCM(const vector<double>& data) {
    vector<int16_t> pcmData;
    pcmData.reserve(data.size());
    for (const auto& val : data) {
        // Convert double to 16-bit integer (PCM format)
        pcmData.push_back(static_cast<int16_t>(val * 32767.0));
    }
    return pcmData;
}

//write intgers in spcific amount of bytes
void writeToFile(ofstream& file, int value, int size)
{
    file.write(reinterpret_cast<const char*> (&value), size);
}

//write audio data to a file
void writeWavFile(vector<double> audioData, string filename)
{
    // You can now use audioData as an array of signed 16-bit integers
    ofstream audioFile;
    audioFile.open(filename, ios::binary);

    //header chunk
    audioFile << "RIFF"; // standard ID of the header chunk (4 bytes)
    audioFile << "----"; // size of the file (unknown yet)
    audioFile << "WAVE"; //data of the header chunck

    //format chunk
    audioFile << "fmt "; // standard ID of the format chunk (4 bytes with space)
    writeToFile(audioFile, 16, 4); // standard size of format chunk is 16 bytes
    writeToFile(audioFile, 1, 2); // compression code 1 = geen compressie
    writeToFile(audioFile, 1, 2); // channels
    writeToFile(audioFile, 8000, 4); // sample rate
    writeToFile(audioFile, 16000, 4); // avg_bytes_per_sec
    writeToFile(audioFile, 2, 2); // block align
    writeToFile(audioFile, 16, 2); //bit depth

    //data chunck
    audioFile << "data";
    audioFile << "----";

    int prepos = audioFile.tellp(); // get current position in file

    //write the collected audio data to the new file
    for (int i = 0; i < audioData.size(); i++) {
        int16_t sample = static_cast<int16_t>(audioData[i] * 32768.0f);
        audioFile.write(reinterpret_cast<const char*>(&sample), sizeof(sample));
    }


    int postpos = audioFile.tellp(); // get current position in file

    audioFile.seekp(prepos - 4); // jump to 4 bytes before prepos and overwrite it with size)
    writeToFile(audioFile, postpos - prepos, 4);

    audioFile.seekp(4, ios::beg); // jump to the beginning of the file, offsetted with 4 bytes
    writeToFile(audioFile, postpos - 8, 4); // postpos is last position (so entire size), substract the 8 bytes of ID and size

    audioFile.close();
}

// Function to write a vector of complex numbers to a text file
void writeComplexVectorToFile(const vector<complex<double>>& data, const string& filename) {
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        cerr << "Error: Unable to open output file: " << filename << endl;
        return;
    }

    for (const auto& val : data) {
        outFile << val.real() << " " << val.imag() << endl;
    }

    outFile.close();
}

// Function to perform the convolution process
vector<double> performConvolution(const std::vector<short>& audioData1, const std::vector<short>& audioData2) {
    // Calculate desired length for padded signals
    size_t desiredLength = audioData1.size() + audioData2.size() - 1;

    // Pad signals with zeros to desired length
    std::vector<short> paddedAudioData1 = zeroPadSignal(audioData1, desiredLength);
    std::vector<short> paddedAudioData2 = zeroPadSignal(audioData2, desiredLength);

    // Perform FFT on padded signals
    std::vector<std::complex<double>> fftResult1 = computeFFT(paddedAudioData1);
    fftResult1 = computeFFT(paddedAudioData1);
    std::vector<std::complex<double>> fftResult2 = computeFFT(paddedAudioData2);

    // Write FFT results to text files
    writeComplexVectorToFile(fftResult1, "fftResult1.txt");
    writeComplexVectorToFile(fftResult2, "fftResult2.txt");

    std::cout << "FFT results written to fftResult1.txt and fftResult2.txt" << std::endl;

    // Multiply both signals
    std::vector<std::complex<double>> multipliedResult = elementwiseMultiply(fftResult1, fftResult2);

    // Write result to text file
    writeComplexVectorToFile(multipliedResult, "multipliedResult.txt");

    std::cout << "Multiplied result written to multipliedResult.txt" << std::endl;

    // Perform IFFT on multiplied frequency signal
    std::vector<double> timeDomainResult = computeIFFT(multipliedResult);

    // Normalize time domain result
    normalizeVector(timeDomainResult);

    // Write to WAV file
    writeWavFile(timeDomainResult, "fftConvOut.wav");

    return timeDomainResult;
}

int main() {
    const char* inputFile1 = "AUDIO2.wav";
    const char* inputFile2 = "IR.wav";

    // Variables to store audio data
    vector<short> audioData1, audioData2;

    // Read audio data from input files
    if (!handleAudioFiles(inputFile1, inputFile2, audioData1, audioData2)) {
        return 1;
    }

    // Display the number of samples read from each file
    std::cout << "Samples in file 1: " << audioData1.size() << std::endl;
    std::cout << "Samples in file 2: " << audioData2.size() << std::endl;

    // Perform convolution
    //vector<double> output = performConvolution(audioData1, audioData2);

    // Perform convolution 100 times and measure the time taken everytime, write the results to a file
    ofstream timeFile;
    timeFile.open("timeResults.txt");
    for (int i = 0; i < 100; i++) {
        cout << "Iteration " << i + 1 << endl;
        auto start = chrono::high_resolution_clock::now();
        vector<double> output = performConvolution(audioData1, audioData2);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        timeFile << elapsed.count() << endl;
    }

    // Write to WAV file
    //writeWavFile(output, "fftConvOut.wav");

    return 0;
}
