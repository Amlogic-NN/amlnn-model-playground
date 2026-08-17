/*
 * Copyright (C) 2024–2025 Amlogic, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "kiss_fftr.h"
#include "model_loader.h"

inline constexpr int kFrameSize = 512;
inline constexpr int kHopSize = 128;
inline constexpr int kFreqBins = kFrameSize / 2 + 1;

inline constexpr int kStateLayers = 2;
inline constexpr int kStateHeads = 128;
inline constexpr int kStatePerHead = 2;
inline constexpr int kStateSize = kStateLayers * kStateHeads * kStatePerHead;

class FFTProcessor {
public:
    FFTProcessor() {
        constexpr float kPi = 3.1415927f;

        for (size_t i = 0; i < window_.size(); ++i) {
            const float phase = 2.0f * kPi * i / (window_.size() - 1);
            window_[i] = std::sqrt(0.5f * (1.0f - std::cos(phase)));
        }
    }

    FFTProcessor(const FFTProcessor&) = delete;
    FFTProcessor& operator=(const FFTProcessor&) = delete;

    void init() {
        if (fft_cfg_ && ifft_cfg_) {
            return;
        }

        std::unique_ptr<kiss_fftr_state, decltype(&free)> new_fft(
            kiss_fftr_alloc(kFrameSize, 0, nullptr, nullptr), &free);
        std::unique_ptr<kiss_fftr_state, decltype(&free)> new_ifft(
            kiss_fftr_alloc(kFrameSize, 1, nullptr, nullptr), &free);
        if (!new_fft || !new_ifft) {
            throw std::runtime_error("Failed to initialize kissfft");
        }

        fft_cfg_ = std::move(new_fft);
        ifft_cfg_ = std::move(new_ifft);
    }

    void apply_window(const std::array<float, kFrameSize>& input,
                      std::array<float, kFrameSize>& windowed_out) const {
        for (size_t i = 0; i < input.size(); ++i) {
            windowed_out[i] = input[i] * window_[i];
        }
    }

    void forward_transform(const std::array<float, kFrameSize>& windowed,
                           std::array<float, kFreqBins>& magnitude_out,
                           std::array<kiss_fft_cpx, kFreqBins>& spectrum_out) const {
        if (!fft_cfg_ || !ifft_cfg_) {
            throw std::runtime_error("FFT processor is not initialized.");
        }
        kiss_fftr(fft_cfg_.get(), windowed.data(), spectrum_out.data());
        for (size_t i = 0; i < magnitude_out.size(); ++i) {
            magnitude_out[i] = std::hypot(spectrum_out[i].r, spectrum_out[i].i);
        }
    }

    void inverse_transform(const std::array<kiss_fft_cpx, kFreqBins>& spectrum,
                           std::array<float, kFrameSize>& time_domain_out) const {
        if (!fft_cfg_ || !ifft_cfg_) {
            throw std::runtime_error("FFT processor is not initialized.");
        }
        kiss_fftri(ifft_cfg_.get(), spectrum.data(), time_domain_out.data());
        for (size_t i = 0; i < time_domain_out.size(); ++i) {
            time_domain_out[i] /= static_cast<float>(kFrameSize);
        }
    }

    const std::array<float, kFrameSize>& window() const {
        return window_;
    }

private:

    std::unique_ptr<kiss_fftr_state, decltype(&free)> fft_cfg_{nullptr, &free};
    std::unique_ptr<kiss_fftr_state, decltype(&free)> ifft_cfg_{nullptr, &free};
    std::array<float, kFrameSize> window_{};
};

struct Model {
    void* ctx = nullptr;

    Model() noexcept = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void init(const std::string& path) {
        if (init_network(path, ctx) != AMLNN_SUCCESS) {
            throw std::runtime_error("Failed to initialize model.");
        }
    }

    ~Model() {
        if (ctx) {
            uninit_network(ctx);
        }
    }

    template <size_t N0, size_t N1, size_t O0, size_t O1>
    void run(const std::array<float, N0>& in0,
             const std::array<float, N1>& in1,
             std::array<float, O0>& out0,
             std::array<float, O1>& out1) const
    {
        if (!ctx) {
            throw std::runtime_error("Model context is not initialized.");
        }

        amlnn_input inputs[2]{};
        inputs[0].index = 0;
        inputs[0].buf = const_cast<float*>(in0.data());
        inputs[0].size = static_cast<uint32_t>(in0.size() * sizeof(float));

        inputs[1].index = 1;
        inputs[1].buf = const_cast<float*>(in1.data());
        inputs[1].size = static_cast<uint32_t>(in1.size() * sizeof(float));

        if (amlnn_inputs_set(ctx, 2, inputs) != AMLNN_SUCCESS) {
            throw std::runtime_error("Failed to set model inputs.");
        }

        if (amlnn_run(ctx, nullptr) != AMLNN_SUCCESS) {
            throw std::runtime_error("Model inference failed.");
        }

        amlnn_output outputs[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            outputs[i].is_float = 1;
            outputs[i].index = i;
        }

        if (amlnn_outputs_get(ctx, 2, outputs) != AMLNN_SUCCESS) {
            throw std::runtime_error("Failed to get model outputs.");
        }

        if (!outputs[0].buf || !outputs[1].buf) {
            throw std::runtime_error("Model inference failed: output is null.");
        }

        const auto out0_count = outputs[0].size / sizeof(float);
        const auto out1_count = outputs[1].size / sizeof(float);
        if (out0_count < out0.size() || out1_count < out1.size()) {
            throw std::runtime_error("Unexpected model output shapes.");
        }

        std::memcpy(out0.data(), outputs[0].buf, out0.size() * sizeof(float));
        std::memcpy(out1.data(), outputs[1].buf, out1.size() * sizeof(float));
    }
};

class DTLN {
public:
    void init(const std::string& model1_path, const std::string& model2_path) {
        fft_.init();
        model1.init(model1_path);
        model2.init(model2_path);

        inBuf.fill(0.0f);
        outBuf.fill(0.0f);
        s1.fill(0.0f);
        s2.fill(0.0f);
    }

    void process(const std::array<float, kHopSize>& input,
                 std::array<float, kHopSize>& output) {
        shift_input(input);

        fft_.apply_window(inBuf, windowed_);
        fft_.forward_transform(windowed_, magnitude_, spectrum_);

        transpose_in(s1, s1_adla_);
        model1.run(magnitude_, s1_adla_, mask_, s1_next_);
        transpose_out(s1_next_, s1);

        for (size_t i = 0; i < spectrum_.size(); ++i) {
            spectrum_[i].r *= mask_[i];
            spectrum_[i].i *= mask_[i];
        }

        fft_.inverse_transform(spectrum_, windowed_);

        transpose_in(s2, s2_adla_);
        model2.run(windowed_, s2_adla_, enhanced_, s2_next_);
        transpose_out(s2_next_, s2);

        overlap_add(enhanced_, output);
    }

private:
    void shift_input(const std::array<float, kHopSize>& input) {
        std::memmove(inBuf.data(), inBuf.data() + kHopSize, (kFrameSize - kHopSize) * sizeof(float));
        std::memcpy(inBuf.data() + kFrameSize - kHopSize, input.data(), kHopSize * sizeof(float));
    }

    void overlap_add(const std::array<float, kFrameSize>& frame,
                     std::array<float, kHopSize>& output) {
        std::memmove(outBuf.data(), outBuf.data() + kHopSize, (kFrameSize - kHopSize) * sizeof(float));
        std::fill(outBuf.end() - kHopSize, outBuf.end(), 0.0f);
        for (size_t i = 0; i < outBuf.size(); ++i) {
            outBuf[i] += frame[i] * fft_.window()[i];
        }
        std::memcpy(output.data(), outBuf.data(), kHopSize * sizeof(float));
    }

    FFTProcessor fft_;
    Model model1;
    Model model2;
    std::array<float, kFrameSize> inBuf{};
    std::array<float, kFrameSize> outBuf{};
    std::array<float, kFreqBins> magnitude_{};
    std::array<kiss_fft_cpx, kFreqBins> spectrum_{};
    std::array<float, kFreqBins> mask_{};
    std::array<float, kStateSize> s1{};
    std::array<float, kStateSize> s2{};
    std::array<float, kStateSize> s1_adla_{};
    std::array<float, kStateSize> s1_next_{};
    std::array<float, kStateSize> s2_adla_{};
    std::array<float, kStateSize> s2_next_{};
    std::array<float, kFrameSize> windowed_{};
    std::array<float, kFrameSize> enhanced_{};

    // Transpose the state from [layer][head][state] layout
    // to [head][layer][state] layout expected by the ADLA model.
    static void transpose_in(const std::array<float, kStateSize>& src,
                             std::array<float, kStateSize>& dst) {
        for (int l = 0; l < kStateLayers; ++l) {
            for (int h = 0; h < kStateHeads; ++h) {
                for (int s = 0; s < kStatePerHead; ++s) {
                    const int src_idx = l * (kStateHeads * kStatePerHead) + h * kStatePerHead + s;
                    const int dst_idx = h * (kStateLayers * kStatePerHead) + l * kStatePerHead + s;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }

    // Inverse transpose: ADLA model output state back into [layer][head][state].
    static void transpose_out(const std::array<float, kStateSize>& src,
                              std::array<float, kStateSize>& dst) {
        for (int l = 0; l < kStateLayers; ++l) {
            for (int h = 0; h < kStateHeads; ++h) {
                for (int s = 0; s < kStatePerHead; ++s) {
                    const int src_idx = h * (kStateLayers * kStatePerHead) + l * kStatePerHead + s;
                    const int dst_idx = l * (kStateHeads * kStatePerHead) + h * kStatePerHead + s;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
};

struct WavHeader {
    char riff[4];
    uint32_t size;
    char wave[4];
    char fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];
    uint32_t data_size;
};

bool validate_wav_header(const WavHeader& header) {
    if (std::memcmp(header.riff, "RIFF", 4) != 0 || std::memcmp(header.wave, "WAVE", 4) != 0) {
        std::cerr << "Unsupported WAV header: missing RIFF/WAVE." << std::endl;
        return false;
    }
    if (std::memcmp(header.fmt, "fmt ", 4) != 0 || std::memcmp(header.data, "data", 4) != 0) {
        std::cerr << "Unsupported WAV header: expected PCM data chunk." << std::endl;
        return false;
    }
    if (header.audio_format != 1) {
        std::cerr << "Unsupported WAV format: only PCM is supported." << std::endl;
        return false;
    }
    if (header.channels != 1) {
        std::cerr << "Unsupported WAV format: only mono audio is supported." << std::endl;
        return false;
    }
    if (header.bits_per_sample != 16) {
        std::cerr << "Unsupported WAV format: only 16-bit samples are supported." << std::endl;
        return false;
    }
    return true;
}

// Convert a normalized float sample in [-1, 1] to 16-bit PCM.
// Clamp to the target int16 range before rounding to avoid overflow.
inline int16_t float_to_int16(float value) {
    const float scaled = std::clamp(value * 32768.0f, -32768.0f, 32767.0f);
    return static_cast<int16_t>(std::lrintf(scaled));
}

size_t read_samples(std::ifstream& fin, std::array<int16_t, kHopSize>& buffer) {
    fin.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(int16_t));
    return static_cast<size_t>(fin.gcount() / sizeof(int16_t));
}

void update_wav_header(std::ofstream& fout, const WavHeader& header, uint32_t data_size) {
    WavHeader updated = header;
    updated.data_size = data_size;
    updated.size = data_size + sizeof(WavHeader) - 8;
    fout.seekp(0, std::ios::beg);
    fout.write(reinterpret_cast<const char*>(&updated), sizeof(updated));
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <model1.adla> <model2.adla> <in.wav> <out.wav>\n";
        return -1;
    }

    const char* model1_path = argv[1];
    const char* model2_path = argv[2];
    const char* input_wav = argv[3];
    const char* output_wav = argv[4];

    std::ifstream fin(input_wav, std::ios::binary);
    if (!fin) {
        std::cerr << "Cannot open input WAV: " << input_wav << std::endl;
        return -1;
    }

    WavHeader header;
    if (!fin.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        std::cerr << "Failed to read WAV header." << std::endl;
        return -1;
    }
    if (!validate_wav_header(header)) {
        return -1;
    }

    std::ofstream fout(output_wav, std::ios::binary);
    if (!fout) {
        std::cerr << "Cannot create output WAV: " << output_wav << std::endl;
        return -1;
    }
    fout.write(reinterpret_cast<const char*>(&header), sizeof(header));

    DTLN dtln;
    try {
        dtln.init(model1_path, model2_path);
    } catch (const std::exception& ex) {
        std::cerr << "DTLN initialization failed: " << ex.what() << std::endl;
        return -1;
    }

    std::array<int16_t, kHopSize> input_block{};
    std::array<float, kHopSize> float_input{};
    std::array<float, kHopSize> float_output{};
    std::array<int16_t, kHopSize> output_block{};

    uint32_t bytes_written = 0;
    while (true) {
        const size_t samples_read = read_samples(fin, input_block);
        if (samples_read == 0) {
            break;
        }

        if (samples_read < input_block.size()) {
            std::fill(input_block.begin() + samples_read, input_block.end(), 0);
        }

        for (size_t i = 0; i < input_block.size(); ++i) {
            float_input[i] = input_block[i] * 1.0f / 32768.0f;
        }

        try {
            dtln.process(float_input, float_output);
        } catch (const std::exception& ex) {
            std::cerr << "DTLN inference failed: " << ex.what() << std::endl;
            return -1;
        }

        for (size_t i = 0; i < output_block.size(); ++i) {
            output_block[i] = float_to_int16(float_output[i]);
        }

        fout.write(reinterpret_cast<const char*>(output_block.data()), output_block.size() * sizeof(int16_t));
        bytes_written += static_cast<uint32_t>(output_block.size() * sizeof(int16_t));

        if (samples_read < input_block.size()) {
            break;
        }
    }

    update_wav_header(fout, header, bytes_written);
    std::cout << "Done. Output: " << output_wav << " (" << bytes_written << " bytes)" << std::endl;
    return 0;
}