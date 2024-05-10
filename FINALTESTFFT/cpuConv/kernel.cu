
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <cmath>

using namespace std;

//defines for stereo
//#define samp_rate 96000
//#define bit_depth 16
//#define ch 2
//#define avg_bytes_per_sec 384000

//defines for mono
//#define samp_rate 96000
//#define bits_per_sample 8
//#define ch 1
//#define avg_bytes_per_sec 96000

//defines for test data 2
#define samp_rate 8000
#define bits_per_sample 16
#define ch 1
#define avg_bytes_per_sec 16000

// Function to read a WAV file and extract audio data
vector<float> readWavFile(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error opening file: " << filename << endl;
        return {};
    }

    // Check if the file starts with the RIFF header
    char header[4];
    file.read(header, 4);
    if (string(header, 4) != "RIFF") {
        cerr << "Not a valid WAV file." << endl;
        return {};
    }

    // Seek to the start of the data chunk (skipping other headers)
    file.seekg(40, ios::beg);

    // Read the audio data
    vector<float> audioData;
    int16_t sample;
    while (file.read(reinterpret_cast<char*>(&sample), sizeof(sample))) {
        // Convert the 16-bit signed integer sample to 16-bit signed floating-point
        float floatSample = static_cast<float>(sample) / 32768.0f; // Normalization
        audioData.push_back(floatSample);
    }

    file.close();
    return audioData;
}

//write intgers in spcific amount of bytes
void writeToFile(ofstream& file, int value, int size)
{
    file.write(reinterpret_cast<const char*> (&value), size);
}

//write audio data to a file
void writeWavFile(vector<float> audioData, string filename)
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
    writeToFile(audioFile, ch, 2); // channels
    writeToFile(audioFile, samp_rate, 4); // sample rate
    writeToFile(audioFile, avg_bytes_per_sec, 4); // avg_bytes_per_sec
    writeToFile(audioFile, 2, 2); // block align
    writeToFile(audioFile, bits_per_sample, 2); //bit depth

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

vector<float> convolution(const vector<float>& signal, const vector<float>& impulse_response) {
    vector<float> output(signal.size() + impulse_response.size() - 1, 0.0f);

    for (size_t i = 0; i < signal.size(); ++i) {
        for (size_t j = 0; j < impulse_response.size(); ++j) {
            output[i + j] += signal[i] * impulse_response[j];
        }
    }

    return output;
}

template<typename T>
vector<T>
conv(std::vector<T> const& f, vector<T> const& g) {
    int const nf = f.size();
    int const ng = g.size();
    int const n = nf + ng - 1;
    vector<T> out(n, T());
    for (auto i(0); i < n; ++i) {
        int const jmn = (i >= ng - 1) ? i - (ng - 1) : 0;
        int const jmx = (i < nf - 1) ? i : nf - 1;
        for (auto j(jmn); j <= jmx; ++j) {
            out[i] += (f[j] * g[i - j]);
        }
    }
    return out;
}

// Function to calculate L2 norm of a vector
float calculateL2Norm(const std::vector<float>& vec) {
    float sum = 0.0;
    for (const auto& value : vec) {
        sum += value * value;
    }
    return sqrt(sum);
}

int main() {
    // Provide the WAV file paths
    string filenameAudio = "AUDIO2.wav";
    string filenameImpulseResponse = "IR.wav";

    // Read the WAV files and extract audio data
    vector<float> audioData = readWavFile(filenameAudio);
    vector<float> irData = readWavFile(filenameImpulseResponse);

    // Check if audio data is extracted successfully
    if (audioData.empty() || irData.empty()) {
        cerr << "Failed to read audio data from files." << endl;
        return 1;
    }

    float irL2Norm = calculateL2Norm(irData);
    vector<float> normalizedIr;
    for (const auto& sample : irData) {
        normalizedIr.push_back(sample / irL2Norm);
    }

    // Print the number of audio samples extracted
    cout << "Number of audio samples in audio: " << audioData.size() << endl;
    cout << "Number of audio samples in impulse response: " << irData.size() << endl;

    // Perform convolution
    //vector<float> output = conv(audioData, normalizedIr);

    // Perform convolution 100 times and measure the time taken everytime, write the results to a file
    ofstream out("output.txt");
    for (int i = 0; i < 100; i++) {
        auto start = chrono::high_resolution_clock::now();
        vector<float> output = conv(audioData, normalizedIr);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        out << elapsed.count() << endl;
    }

    // Write the output to a WAV file
    //writeWavFile(out, "OUTPUT_TEST_28.wav");

    return 0;
}
