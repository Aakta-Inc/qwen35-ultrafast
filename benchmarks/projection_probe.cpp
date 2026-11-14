// SPDX-License-Identifier: Apache-2.0
// Probe real GGUF projection shapes. Output dumps exclude timings for exact comparison.
#include "ggml-cpp.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static std::vector<char> weights(std::ifstream &file, const gguf_context *meta,
                                 int64_t index, size_t size) {
    std::vector<char> bytes(size);
    file.seekg(gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, index));
    file.read(bytes.data(), bytes.size());
    require(bool(file), "cannot read tensor weights");
    return bytes;
}

static int compare(const char *left, const char *right) {
    std::ifstream a(left, std::ios::binary), b(right, std::ios::binary);
    require(bool(a) && bool(b), "cannot open comparison dumps");
    std::string ah, bh;
    std::getline(a, ah); std::getline(b, bh);
    require(ah == "QWEN-PROJECTION-PROBE-v2" && ah == bh, "invalid dump version");
    size_t count = 0, different = 0, cases = 0;
    double max_abs = 0, max_scaled = 0;
    while (std::getline(a, ah) && std::getline(b, bh)) {
        require(ah == bh, "projection contracts differ");
        if (ah.rfind("COMPLETE ", 0) == 0) {
            require(cases == std::stoull(ah.substr(9)) && cases > 0, "invalid completed case count");
            require(a.peek() == EOF && b.peek() == EOF, "trailing dump data");
            const bool pass = max_scaled <= 2e-5;
            std::cout << "cases=" << cases << " values=" << count << " bit_differences=" << different
                      << " max_abs=" << max_abs << " max_abs_over_one_plus_reference=" << max_scaled
                      << " tolerance=2e-5 pass=" << pass << '\n';
            return pass ? 0 : 2;
        }
        const auto first = ah.find(','), second = ah.find(',', first + 1), third = ah.find(',', second + 1);
        const auto fourth = ah.find(',', third + 1);
        const auto fifth = ah.find(',', fourth + 1);
        require(third != std::string::npos && fourth != std::string::npos && fifth != std::string::npos, "invalid dump header");
        const size_t rows = std::stoull(ah.substr(third + 1, fourth - third - 1));
        const size_t columns = std::stoull(ah.substr(fourth + 1, fifth - fourth - 1));
        require(rows > 0 && rows <= 1000000, "invalid dump row count");
        require(columns > 0 && columns <= 512, "invalid dump column count");
        const size_t values = rows * columns;
        std::vector<float> av(values), bv(values);
        a.read(reinterpret_cast<char *>(av.data()), values * sizeof(float));
        b.read(reinterpret_cast<char *>(bv.data()), values * sizeof(float));
        require(bool(a) && bool(b), "truncated output dump");
        for (size_t i = 0; i < values; ++i) {
            require(std::isfinite(av[i]) && std::isfinite(bv[i]), "non-finite comparison value");
            different += std::memcmp(&av[i], &bv[i], sizeof(float)) != 0;
            const double error = std::abs(double(av[i]) - bv[i]);
            max_abs = std::max(max_abs, error);
            max_scaled = std::max(max_scaled, error / (1.0 + std::abs(double(av[i]))));
        }
        count += values;
        ++cases;
    }
    throw std::runtime_error("incomplete output dump");
}

int main(int argc, char **argv) try {
    if (argc == 4 && std::string(argv[1]) == "--compare") return compare(argv[2], argv[3]);
    require(argc >= 3, "usage: projection-probe MODEL.gguf NEW_OUTPUT.bin [--check] [--columns N] | --compare REFERENCE.bin CANDIDATE.bin");
    bool check_only = false;
    int columns = 1;
    for (int i = 3; i < argc; ++i) {
        const std::string option(argv[i]);
        if (option == "--check") check_only = true;
        else if (option == "--columns" && i + 1 < argc) {
            const std::string value(argv[++i]);
            size_t used = 0;
            columns = std::stoi(value, &used);
            require(used == value.size() && columns > 0 && columns <= 512, "columns must be 1..512");
        } else throw std::runtime_error("unknown or incomplete option");
    }
    require(!std::ifstream(argv[2]).good(), "output already exists");
    std::ifstream file(argv[1], std::ios::binary);
    require(bool(file), "cannot open model");
    ggml_context *raw = nullptr;
    gguf_context_ptr meta(gguf_init_from_file(argv[1], {true, &raw}));
    ggml_context_ptr tensors(raw);
    require(bool(meta) && bool(tensors), "invalid GGUF metadata");
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));
    require(bool(backend), "CUDA initialization failed");
    std::ofstream dump(argv[2], std::ios::binary);
    require(bool(dump), "cannot create output dump");
    dump << "QWEN-PROJECTION-PROBE-v2\n";
    std::cout << "tensor,type,k,rows,columns,fused,seed,median_us\n";
    std::set<std::tuple<int, int64_t, int64_t, bool>> seen;
    size_t cases = 0;
    for (int64_t index = 0; index < gguf_get_n_tensors(meta.get()); ++index) {
        const char *name = gguf_get_tensor_name(meta.get(), index);
        const auto *source = ggml_get_tensor(tensors.get(), name);
        const auto type = source->type;
        if ((type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q5_K && type != GGML_TYPE_Q6_K) ||
            source->ne[2] != 1 || source->ne[3] != 1 || source->ne[1] <= 1) continue;
        const int64_t k = source->ne[0], rows = source->ne[1];
        // The output head is evaluated only at selected logits positions, not every image token.
        if (columns > 1 && (std::string(name) == "token_embd.weight" || std::string(name) == "output.weight")) continue;
        for (bool fused : {false, true}) {
            std::string gate_name(name);
            const auto up = gate_name.find("ffn_up.weight");
            if (fused && up == std::string::npos) continue;
            if (fused) gate_name.replace(up, std::string("ffn_up.weight").size(), "ffn_gate.weight");
            const int64_t gate_index = fused ? gguf_find_tensor(meta.get(), gate_name.c_str()) : -1;
            const auto *gate_source = fused ? ggml_get_tensor(tensors.get(), gate_name.c_str()) : nullptr;
            if (fused && (!gate_source || gate_index < 0 || gate_source->type != type ||
                          gate_source->ne[0] != k || gate_source->ne[1] != rows)) continue;
            if (!seen.emplace(type, k, rows, fused).second) continue;

            ggml_context_ptr ctx(ggml_init({2 * 1024 * 1024, nullptr, true}));
            require(bool(ctx), "graph metadata allocation failed");
            auto *w = ggml_new_tensor_2d(ctx.get(), type, k, rows);
            auto *x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, columns);
            ggml_set_input(w);
            ggml_set_input(x);
            auto *result = ggml_mul_mat(ctx.get(), w, x);
            ggml_tensor *gate = nullptr;
            if (fused) {
                gate = ggml_new_tensor_2d(ctx.get(), type, k, rows);
                ggml_set_input(gate);
                result = ggml_swiglu_split(ctx.get(), ggml_mul_mat(ctx.get(), gate, x), result);
            }
            ggml_set_output(result);
            auto *graph = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(graph, result);
            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
            require(bool(buffer), "GPU allocation failed");
            auto bytes = weights(file, meta.get(), index, ggml_nbytes(w));
            ggml_backend_tensor_set(w, bytes.data(), 0, bytes.size());
            if (gate) {
                auto gate_bytes = weights(file, meta.get(), gate_index, ggml_nbytes(gate));
                ggml_backend_tensor_set(gate, gate_bytes.data(), 0, gate_bytes.size());
            }
            auto compute = [&] {
                require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS,
                        "GPU graph execution failed");
            };
            for (uint32_t seed = 0; seed < 3; ++seed) {
                std::vector<float> input(k * columns), output(rows * columns);
                uint32_t state = seed + 17;
                for (auto &value : input) {
                    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                    value = seed == 0 ? 0.f : (float(state & 65535) / 32768.f - 1.f);
                }
                ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
                for (int warmup = 0; warmup < (check_only ? 1 : 10); ++warmup) compute();
                ggml_backend_synchronize(backend.get());
                std::vector<double> times;
                for (int round = 0; round < (check_only ? 0 : 3); ++round) {
                    const auto start = std::chrono::steady_clock::now();
                    for (int repeat = 0; repeat < 100; ++repeat) compute();
                    ggml_backend_synchronize(backend.get());
                    times.push_back(std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - start).count() / 100);
                }
                ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
                for (float value : output) require(std::isfinite(value), "non-finite projection output");
                std::sort(times.begin(), times.end());
                std::cout << name << ',' << ggml_type_name(type) << ',' << k << ',' << rows << ','
                          << columns << ',' << fused << ',' << seed << ',' << (check_only ? 0 : times[1]) << '\n' << std::flush;
                dump << name << ',' << type << ',' << k << ',' << rows << ',' << columns << ',' << fused << ',' << seed << '\n';
                dump.write(reinterpret_cast<const char *>(output.data()), output.size() * sizeof(float));
                require(bool(dump), "output write failed");
                ++cases;
            }
        }
    }
    require(cases > 0, "model contains no supported projections");
    dump << "COMPLETE " << cases << '\n';
    dump.close();
    require(bool(dump), "output close failed");
    std::cerr << "Completed " << cases << " projection cases; compare full dumps for parity.\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
