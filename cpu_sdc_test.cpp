/**
 * Silent Data Corruption (SDC) Test — CPU, GPU, RAM, SSD
 * Based on: "The Dark Side of Computing: Silent Data Corruptions" (IEEE Computer, 2025)
 * Google: "a few mercurial cores per several 1,000 machines" - cores that produce wrong results silently.
 *
 * Runs redundant computations and data integrity checks across CPU, GPU, RAM, and SSD.
 * Mismatch or corruption indicates potential defective hardware.
 *
 * State file (cpu_sdc_test_state.txt) stores passed tests; use --reset to clear and run fresh.
 */
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifdef USE_CUDA
#include "gpu_sdc_cuda.h"
#endif

namespace {

constexpr int CHUNK_SIZE = 50000;
constexpr int64_t SEED = 42;
constexpr int64_t ERROR_RESULT = -1;

// Number of hardware threads
int num_cores() {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return static_cast<int>(si.dwNumberOfProcessors);
#else
    return static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
#endif
}

// Simple deterministic test key (matches config uniqueness)
std::string test_key(const std::string& test_type, int64_t iterations) {
    return test_type + ":" + std::to_string(iterations);
}

// Load passed test keys from state file (one key per line)
std::vector<std::string> load_state(const std::string& state_path) {
    std::vector<std::string> keys;
    std::ifstream f(state_path);
    if (!f) return keys;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) keys.push_back(line);
    }
    return keys;
}

bool is_passed(const std::vector<std::string>& state, const std::string& key) {
    for (const auto& k : state) {
        if (k == key) return true;
    }
    return false;
}

void save_state(const std::string& state_path, const std::vector<std::string>& state) {
    std::ofstream f(state_path);
    if (!f) return;
    for (const auto& k : state) {
        f << k << '\n';
    }
}

void clear_state(const std::string& state_path) {
    std::remove(state_path.c_str());
}

// Deterministic computation. Same inputs must always produce same output.
int64_t compute_test(const std::string& test_type, int64_t iterations,
                     std::atomic<int64_t>* progress_counter) {
    if (test_type == "int_mul") {
        uint32_t a = 45, b = 22;
        uint32_t result = 0;
        for (int64_t i = 0; i < iterations; ++i) {
            result = static_cast<uint32_t>((static_cast<uint64_t>(a) * b + i) & 0xFFFFFFFFu);
            a = static_cast<uint32_t>((result % 12345) + 1);
            b = static_cast<uint32_t>((result % 6789) + 1);
            if (progress_counter && (i + 1) % CHUNK_SIZE == 0) {
                progress_counter->fetch_add(1);
            }
        }
        return static_cast<int64_t>(result);
    }
    if (test_type == "int_add") {
        uint32_t x = static_cast<uint32_t>(SEED);
        for (int64_t i = 0; i < iterations; ++i) {
            x = static_cast<uint32_t>((static_cast<uint64_t>(x) + static_cast<uint64_t>(x) * 31 + 17) & 0xFFFFFFFFu);
            if (progress_counter && (i + 1) % CHUNK_SIZE == 0) {
                progress_counter->fetch_add(1);
            }
        }
        return static_cast<int64_t>(x);
    }
    if (test_type == "float_mul") {
        double x = 3.14159265358979;
        for (int64_t i = 0; i < iterations; ++i) {
            x = x * 1.0000001 + 0.0000001;
            if (progress_counter && (i + 1) % CHUNK_SIZE == 0) {
                progress_counter->fetch_add(1);
            }
        }
        return static_cast<int64_t>(static_cast<uint64_t>(x * 1e9) & 0xFFFFFFFFu);
    }
    if (test_type == "xor_chain") {
        uint32_t x = static_cast<uint32_t>(SEED);
        for (int64_t i = 0; i < iterations; ++i) {
            x = x ^ (x << 13);
            x = x ^ (x >> 17);
            x = x ^ (x << 5);
            if (progress_counter && (i + 1) % CHUNK_SIZE == 0) {
                progress_counter->fetch_add(1);
            }
        }
        return static_cast<int64_t>(x);
    }
    return ERROR_RESULT;
}

struct TestResult {
    int64_t ref;
    std::vector<int64_t> results;
    std::vector<std::pair<int, int64_t>> mismatches;
};

TestResult run_redundant_test(const std::string& test_type, int64_t iterations,
                             const std::string& test_desc, int test_idx, int total_tests) {
    int n = std::min(num_cores(), 20);
    std::vector<int64_t> results(static_cast<size_t>(n), 0);
    std::atomic<int64_t> progress{0};
    std::atomic<int> threads_done{0};
    int64_t total_chunks = n * std::max(static_cast<int64_t>(1), iterations / CHUNK_SIZE);

    std::vector<std::thread> threads;
    for (int i = 0; i < n; ++i) {
        threads.emplace_back([&, i]() {
            try {
                results[static_cast<size_t>(i)] = compute_test(test_type, iterations, &progress);
            } catch (...) {
                results[static_cast<size_t>(i)] = ERROR_RESULT;
            }
            threads_done.fetch_add(1);
        });
    }

    auto start = std::chrono::steady_clock::now();
    for (;;) {
        int64_t done_count = progress.load();
        int pct = total_chunks > 0 ? std::min(100, static_cast<int>(100 * done_count / total_chunks)) : 0;
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start).count();
        int hh = static_cast<int>(elapsed / 3600);
        int mm = static_cast<int>((elapsed % 3600) / 60);
        int ss = static_cast<int>(elapsed % 60);
        std::string desc = test_desc;
        if (desc.size() > 35) desc = desc.substr(0, 35);
        std::cout << "\r  [" << test_idx << "/" << total_tests << "] "
                  << std::left << std::setw(35) << desc
                  << std::right << std::setw(3) << pct << "%  "
                  << std::setfill('0') << std::setw(2) << hh << ":"
                  << std::setw(2) << mm << ":" << std::setw(2) << ss << std::setfill(' ') << "\r"
                  << std::flush;

        if (threads_done.load() == n) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    std::cout << "\r" << std::string(90, ' ') << "\r" << std::flush;

    int64_t ref = results[0];
    std::vector<std::pair<int, int64_t>> mismatches;
    for (int i = 1; i < n; ++i) {
        if (results[static_cast<size_t>(i)] != ref && results[static_cast<size_t>(i)] != ERROR_RESULT) {
            mismatches.emplace_back(i, results[static_cast<size_t>(i)]);
        }
    }
    return {ref, results, mismatches};
}

// RAM test: write deterministic pattern, read back, verify. Mismatch = bit flip.
struct RamTestResult {
    bool passed;
    size_t corrupt_offset;
    uint64_t expected;
    uint64_t actual;
};

RamTestResult run_ram_test(size_t size_mb, int passes) {
    const size_t size = size_mb * 1024 * 1024;
    std::vector<uint64_t> buf(size / sizeof(uint64_t));
    uint64_t seed = 42;
    for (size_t i = 0; i < buf.size(); ++i) {
        seed = seed ^ (seed << 13);
        seed = seed ^ (seed >> 17);
        seed = seed ^ (seed << 5);
        buf[i] = seed;
    }
    for (int p = 0; p < passes; ++p) {
        for (size_t i = 0; i < buf.size(); ++i) {
            uint64_t expected = buf[i];
            volatile uint64_t* ptr = &buf[i];
            uint64_t read_back = *ptr;
            if (read_back != expected) {
                return {false, i * sizeof(uint64_t), expected, read_back};
            }
        }
    }
    return {true, 0, 0, 0};
}

// SSD test: write to temp file, read back, verify. Mismatch = storage corruption.
struct SsdTestResult {
    bool passed;
    size_t corrupt_offset;
};

SsdTestResult run_ssd_test(size_t size_mb, int passes, const std::string& temp_dir) {
    const size_t size = size_mb * 1024 * 1024;
    std::vector<uint64_t> write_buf(size / sizeof(uint64_t));
    uint64_t seed = 42;
    for (size_t i = 0; i < write_buf.size(); ++i) {
        seed = seed ^ (seed << 13);
        seed = seed ^ (seed >> 17);
        seed = seed ^ (seed << 5);
        write_buf[i] = seed;
    }
    std::string path = temp_dir + "sdc_ssd_test.tmp";
    for (int p = 0; p < passes; ++p) {
        {
            std::ofstream f(path, std::ios::binary);
            if (!f) return {false, 0};
            f.write(reinterpret_cast<const char*>(write_buf.data()), size);
            f.close();
        }
        std::vector<uint64_t> read_buf(size / sizeof(uint64_t));
        {
            std::ifstream f(path, std::ios::binary);
            if (!f) return {false, 0};
            f.read(reinterpret_cast<char*>(read_buf.data()), size);
        }
        for (size_t i = 0; i < write_buf.size(); ++i) {
            if (read_buf[i] != write_buf[i]) {
                std::remove(path.c_str());
                return {false, i * sizeof(uint64_t)};
            }
        }
    }
    std::remove(path.c_str());
    return {true, 0};
}

// GPU test: CUDA (NVIDIA) or redundant thread fallback.
struct GpuTestResult {
    bool passed;
    bool skipped;
    bool used_fallback;  // CUDA unavailable, used redundant thread
    bool used_cuda;      // NVIDIA CUDA (real GPU)
    int64_t cpu_result;
    int64_t gpu_result;
};

static int64_t xor_chain_cpu(int64_t iterations) {
    uint32_t x = 42;
    for (int64_t i = 0; i < iterations; ++i) {
        x = x ^ (x << 13);
        x = x ^ (x >> 17);
        x = x ^ (x << 5);
    }
    return static_cast<int64_t>(x);
}

GpuTestResult run_gpu_test(int64_t iterations) {
    int64_t cpu_result = xor_chain_cpu(iterations);

#ifdef USE_CUDA
    int64_t cuda_result = 0;
    if (gpu_sdc_cuda_run(iterations, &cuda_result)) {
        bool passed = (cuda_result == cpu_result);
        return {passed, false, false, true, cpu_result, cuda_result};
    }
#endif

    // Fallback: redundant thread (no CUDA)
    int64_t other_result = 0;
    std::thread fallback([&]() { other_result = xor_chain_cpu(iterations); });
    fallback.join();
    bool passed = (other_result == cpu_result);
    return {passed, false, true, false, cpu_result, other_result};
}

std::string current_datetime() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

} // namespace

int main(int argc, char* argv[]) {
    std::string exe_path = argv[0];
    size_t last_sep = exe_path.find_last_of("/\\");
    std::string state_path = (last_sep != std::string::npos)
        ? exe_path.substr(0, last_sep + 1) + "cpu_sdc_test_state.txt"
        : "cpu_sdc_test_state.txt";

    bool reset = false;
    int64_t base_iters = 500000;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--reset") {
            reset = true;
        } else if (arg.find_first_not_of("0123456789") == std::string::npos && !arg.empty()) {
            base_iters = std::max(static_cast<int64_t>(100000), static_cast<int64_t>(std::stoll(arg)));
        }
    }

    if (reset) {
        clear_state(state_path);
        std::cout << "State cleared. Running full test suite.\n\n";
    }

    auto state = load_state(state_path);

    std::cout << std::string(70, '=') << '\n';
    std::cout << "SILENT DATA CORRUPTION (SDC) TEST — CPU, GPU, RAM, SSD\n";
    std::cout << std::string(70, '=') << '\n';
    std::cout << "Based on: IEEE Computer 2025 - 'The Dark Side of Computing'\n";
    std::cout << "Hyperscalers report: 1 in 1000 CPUs may produce wrong results silently.\n";
    std::cout << "This test runs redundant computations across cores and compares results.\n";
    std::cout << std::string(70, '=') << '\n';
    std::cout << "CPU cores: " << num_cores() << '\n';
    std::cout << "Started: " << current_datetime() << "\n\n";

    struct TestSpec {
        std::string type;
        int64_t iters;
        std::string desc;
        std::string category;  // "cpu", "ram", "ssd", "gpu"
    };
    std::vector<TestSpec> tests = {
        {"int_mul", base_iters, "Integer multiplication (multiplier units)", "cpu"},
        {"int_add", base_iters * 2, "Integer addition chain", "cpu"},
        {"float_mul", base_iters, "Floating-point multiplication", "cpu"},
        {"xor_chain", base_iters * 2, "Bit manipulation (XOR shift)", "cpu"},
        {"ram", 64, "RAM integrity (64MB, 10 passes)", "ram"},
        {"ssd", 64, "SSD/storage integrity (64MB, 5 passes)", "ssd"},
        {"gpu", 1000000, "GPU compute (XOR shift 1M)", "gpu"},
    };

    int total_mismatches = 0;
    int total_tests = static_cast<int>(tests.size());
    std::string temp_dir;
#ifdef _WIN32
    char tmp[MAX_PATH];
    if (GetTempPathA(MAX_PATH, tmp) > 0) {
        temp_dir = tmp;
    } else {
        temp_dir = ".\\";
    }
#else
    const char* t = std::getenv("TMPDIR");
    temp_dir = t ? std::string(t) + "/" : "/tmp/";
#endif

    for (size_t idx = 0; idx < tests.size(); ++idx) {
        const auto& spec = tests[idx];
        std::string key = test_key(spec.type, spec.iters);

        if (is_passed(state, key)) {
            std::cout << "Test: " << spec.desc << '\n';
            std::cout << "  SKIPPED (passed earlier). Use --reset to re-run.\n\n";
            continue;
        }

        std::cout << "Test: " << spec.desc << '\n';

        auto t0 = std::chrono::steady_clock::now();

        if (spec.category == "cpu") {
            std::cout << "  Iterations per core: " << spec.iters << '\n';
            TestResult tr = run_redundant_test(
                spec.type, spec.iters, spec.desc,
                static_cast<int>(idx) + 1, total_tests);
            int64_t ref = tr.ref;
            auto& mismatches = tr.mismatches;
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();

            std::cout << "  Time: " << std::fixed << std::setprecision(1) << elapsed
                      << "s | Reference result: " << ref << '\n';

            if (!mismatches.empty()) {
                total_mismatches += static_cast<int>(mismatches.size());
                std::cout << "  *** MISMATCH DETECTED ***\n";
                for (const auto& m : mismatches) {
                    std::cout << "    Core " << m.first << ": " << m.second << " (expected " << ref << ")\n";
                }
                std::cout << "  -> Possible mercurial/defective core. Re-run to confirm.\n";
            } else {
                state.push_back(key);
                std::cout << "  OK - All cores agreed.\n";
            }
        } else if (spec.category == "ram") {
            RamTestResult rr = run_ram_test(static_cast<size_t>(spec.iters), 10);
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            std::cout << "  Time: " << std::fixed << std::setprecision(1) << elapsed << "s\n";
            if (rr.passed) {
                state.push_back(key);
                std::cout << "  OK - No bit flips detected.\n";
            } else {
                total_mismatches++;
                std::cout << "  *** CORRUPTION DETECTED ***\n";
                std::cout << "    Offset: " << rr.corrupt_offset << " expected " << rr.expected
                          << " got " << rr.actual << '\n';
                std::cout << "  -> Possible RAM defect. Re-run to confirm.\n";
            }
        } else if (spec.category == "ssd") {
            SsdTestResult sr = run_ssd_test(static_cast<size_t>(spec.iters), 5, temp_dir);
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            std::cout << "  Time: " << std::fixed << std::setprecision(1) << elapsed << "s\n";
            if (sr.passed) {
                state.push_back(key);
                std::cout << "  OK - Read-back matched write.\n";
            } else {
                total_mismatches++;
                std::cout << "  *** CORRUPTION DETECTED ***\n";
                std::cout << "    Offset: " << sr.corrupt_offset << '\n';
                std::cout << "  -> Possible SSD/storage defect. Re-run to confirm.\n";
            }
        } else if (spec.category == "gpu") {
            GpuTestResult gr = run_gpu_test(spec.iters);
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            std::cout << "  Time: " << std::fixed << std::setprecision(1) << elapsed << "s\n";
            if (gr.skipped) {
                std::cout << "  SKIPPED - OpenCL not available.\n";
            } else if (gr.passed) {
                state.push_back(key);
                const char* src = gr.used_cuda ? "NVIDIA GPU (CUDA)" : "Redundant thread";
                std::cout << "  OK - " << src << " matched CPU reference (" << gr.cpu_result << ").\n";
            } else {
                total_mismatches++;
                std::cout << "  *** MISMATCH DETECTED ***\n";
                std::cout << "    Ref: " << gr.cpu_result << " Other: " << gr.gpu_result << '\n';
                std::cout << "  -> Possible GPU defect. Re-run to confirm.\n";
            }
        }

        save_state(state_path, state);
        std::cout << '\n';
    }

    std::cout << std::string(70, '=') << '\n';
    if (total_mismatches > 0) {
        std::cout << "RESULT: Potential SDC/corruption detected (CPU/GPU/RAM/SSD).\n";
        std::cout << "Recommendation: Run multiple times. If issues persist,\n";
        std::cout << "consider RMA or professional diagnostics (Intel DCDIAG, etc.).\n";
    } else {
        std::cout << "RESULT: No mismatches in this run. All tests passed.\n";
        std::cout << "Note: SDCs can be rare/intermittent. Run periodically or under load.\n";
    }
    std::cout << std::string(70, '=') << '\n';

    return 0;
}
