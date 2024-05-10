#include <iostream>
#include <fstream>
#include <Windows.h>
#include <mmsystem.h>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>

#pragma comment(lib, "Winmm.lib")
using namespace std;

//writeToCSV
//help function to write stuff to csv
void writeRecordToFile(std::string filename, std::string fieldOne, std::string fieldTwo, int fieldThree)
{
    std::ofstream file;
    file.open(filename, std::ios_base::app);
    file << fieldOne << "," << fieldTwo << "," << fieldThree << std::endl;
    file.close();
}

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

// CUDA kernel to perform element-wise multiplication of two vectors of complex numbers
__global__ void elementwiseMultiplyKernel(const cuDoubleComplex* vec1, const cuDoubleComplex* vec2, cuDoubleComplex* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = cuCmul(vec1[idx], vec2[idx]);
    }
}

//cecear = CCR = Cufft Convolutional Reverb
vector<double> cecear(vector<short>& vec1, vector<short>& vec2) {

    // Check if the input vectors have the same size
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Input vectors must have the same size for element-wise multiplication.");
    }

    size_t size = vec1.size();

    // Convert input vectors to double
    vector<double> doubleVec1(vec1.begin(), vec1.end());
    vector<double> doubleVec2(vec2.begin(), vec2.end());

    // Allocate memory on device for input vectors
    double* d_vec1, * d_vec2;
    cudaMalloc((void**)&d_vec1, sizeof(double) * size);
    cudaMalloc((void**)&d_vec2, sizeof(double) * size);

    // Copy input vectors from host to device
    cudaMemcpy(d_vec1, doubleVec1.data(), sizeof(double) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, doubleVec2.data(), sizeof(double) * size, cudaMemcpyHostToDevice);

    // Perform FFT on input vectors
    cufftHandle fftPlan;
    cufftPlan1d(&fftPlan, size, CUFFT_D2Z, 1); // D2Z transform, forward direction

    cuDoubleComplex* d_fftVec1, * d_fftVec2;
    cudaMalloc((void**)&d_fftVec1, sizeof(cuDoubleComplex) * (size / 2 + 1));
    cudaMalloc((void**)&d_fftVec2, sizeof(cuDoubleComplex) * (size / 2 + 1));

    cufftExecD2Z(fftPlan, d_vec1, d_fftVec1);
    cufftExecD2Z(fftPlan, d_vec2, d_fftVec2);

    // Perform element-wise multiplication on GPU
    cuDoubleComplex* d_result;
    cudaMalloc((void**)&d_result, sizeof(cuDoubleComplex) * (size / 2 + 1));

    int blockSize = 256;
    int numBlocks = ((size / 2) + blockSize - 1) / blockSize;

    elementwiseMultiplyKernel << <numBlocks, blockSize >> > (d_fftVec1, d_fftVec2, d_result, size / 2 + 1);

    // Allocate memory for inverse FFT output on GPU
    double* d_ifftResult;
    cudaMalloc(&d_ifftResult, sizeof(double) * size);

    // Create cuFFT plan for inverse FFT
    cufftHandle ifftPlan;
    cufftPlan1d(&ifftPlan, size, CUFFT_Z2D, 1); // Z2D transform, inverse direction

    // Execute inverse FFT
    cufftExecZ2D(ifftPlan, d_result, d_ifftResult);

    // Copy result back to host
    vector<double> ifftResult(size);
    cudaMemcpy(ifftResult.data(), d_ifftResult, sizeof(double) * size, cudaMemcpyDeviceToHost);

    // Clean up
    cufftDestroy(fftPlan);
    cufftDestroy(ifftPlan);
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_fftVec1);
    cudaFree(d_fftVec2);
    cudaFree(d_result);
    cudaFree(d_ifftResult);

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

// Function to perform the convolution process
vector<double> performConvolution(const vector<short>& audioData1, const vector<short>& audioData2) {
    // Calculate desired length for padded signals
    size_t desiredLength = audioData1.size() + audioData2.size() - 1;

    // Pad signals with zeros to desired length
    vector<short> paddedAudioData1 = zeroPadSignal(audioData1, desiredLength);
    vector<short> paddedAudioData2 = zeroPadSignal(audioData2, desiredLength);

    //cecear = CCR = Cufft Convolutional Reverb (I thought of that)
    vector<double> timeDomainResult = cecear(paddedAudioData1, paddedAudioData2);

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
    cout << "Samples in file 1: " << audioData1.size() << endl;
    cout << "Samples in file 2: " << audioData2.size() << endl;

    // Perform convolution
    //vector<double> output = performConvolution(audioData1, audioData2);

    // Perform convolution 100 times and measure the time taken everytime, write the results to a file
    ofstream timeFile;
    timeFile.open("timeResults.txt");
    for (int i = 0; i < 100; i++) {
        cout << "Iteration " << i + 1 << endl;
        LARGE_INTEGER frequency, start, end;
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&start);
        vector<double> output = performConvolution(audioData1, audioData2);
        QueryPerformanceCounter(&end);
        double elapsed = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
        timeFile << elapsed << endl;
    }

    // Write to WAV file
    //writeWavFile(output, "fftConvOut.wav");

    return 0;
}
