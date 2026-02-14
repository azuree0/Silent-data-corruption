# Prior

**Install**

- **CMake 3.14+** —         https://cmake.org/download/
- **C++17 compiler** —      https://visualstudio.microsoft.com/downloads/
- **NVIDIA CUDA Toolkit** — https://developer.nvidia.com/cuda-downloads

**Build**

```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

**Run**

```powershell
.\build\Release\run.exe
```

# Function

**Explain:** Checks if your CPU has mercurial cores—cores that occasionally produce incorrect computation results without crash or error. A mismatch indicates a possible defective core.

**Cause** Silent data corruption is when a program runs on defective hardware, finishes without crashing or raising an error, but the result is wrong—and nothing in the system reports it.

The program thinks it succeeded; the user gets a bad answer. The "silent" part means no exception, no log, no visible failure. Causes include silicon defects (from manufacturing or aging), cosmic-ray bit flips, or marginal circuits. 

**Hyperscalers** are companies that run very large data centers with huge numbers of servers and CPUs. Hyperscale = infrastructure built to scale to 1,000,000s of servers and 1,000,000,000s of users. 

Examples: Microsoft (Azure), Google, Alibaba, Tencent, Meta (Facebook), Amazon (AWS)

They matter for SDC because: (1) **Scale** — they run 100,000s or 1,000,000s of CPUs; even rare defects (e.g. 1 in 1,000) show up often enough to be a real problem. 

(2) **Data** — they publish studies on hardware reliability: Meta (arXiv): many defective CPUs across large fleets; Google (HotOS): "a few mercurial cores per several 1,000 machines"; Alibaba (SOSP 2023): ~3.61% of CPUs in their fleet linked to SDC. 

(3) **Mitigation** — they use tools like Intel DCDIAG and OpenDCDiag to screen CPUs before and during use. 

(4) **Research** — they fund and drive work on SDC (e.g. Meta's SDC RFP, OCP Server Resilience).

Hyperscalers report that roughly 1 in 1000 CPUs can produce SDCs. This test looks for that kind of defect by running the same computation on every core and comparing results; if 1 core disagrees, it may be defective (a "mercurial" core).

**Solution** Hyperscalers now screen fleets for defective chips. Intel DCDIAG, OpenDCDiag, and similar tools perform in-production and out-of-production testing. This script brings redundant-execution detection to consumer hardware.

**Lower degradation %**

| Factor | Action |
|--------|--------|
| Thermal | Good cooling, avoid sustained high temps |
| Voltage | Avoid overclocking, use stable power |
| Load    | Avoid constant 100% load if possible |

**How the test works**

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  CPU — REDUNDANT EXECUTION                                                  │
│  • Same deterministic computation on every CPU core in parallel             │
│  • Same inputs (seed 42) → same expected output                             │
│  • Each core writes result to shared memory                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GPU — CUDA                                                                 │
│  • NVIDIA: CUDA kernel (XOR shift); compare with CPU reference              │
│  • Fallback: redundant thread if CUDA unavailable                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  RAM — INTEGRITY CHECK                                                      │
│  • Write deterministic pattern to memory; read back; verify                 │
│  • Mismatch → bit flip (cosmic ray, defective cell)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SSD — STORAGE INTEGRITY                                                    │
│  • Write known data to temp file; read back; compare                        │
│  • Mismatch → storage/controller corruption                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  TESTS PERFORMED                                                             │
│  • CPU: int mul (500K), int add (1M), float mul (500K), XOR shift (1M)       │
│  • GPU: XOR shift 1M (CUDA; compare with CPU)                                │
│  • RAM: 64MB pattern, 10 passes                                              │
│  • SSD: 64MB write/read, 5 passes                                            │
│                                                                              │
│  SDCs rare/intermittent; run monthly.                                        │
│  After changes: after BIOS/firmware updates or hardware changes              │
│  Under load: run while stressed (heavy workload) to trigger marginal defects │
└──────────────────────────────────────────────────────────────────────────────┘                                                                            
```

# History

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  2008 — DSN PANEL                                                           │
│  • "Silent data corruption — Myth or reality?"                              │
│  • Industry unsure if SDC exists at scale                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2021 — META, GOOGLE                                                        │
│  • Meta (arXiv): 100s of defective CPUs across 100,000s of machines         │
│  • Google (HotOS): a few mercurial cores per several 1,000 machines         │
└─────────────────────────────────────────────────────────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2022 — META RFP                                                            │
│  • SDC Request for Proposals; academia engagement                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2023 — ALIBABA (SOSP)                                                      │
│  • 3.61% of CPUs in fleet identified as SDC-causing                         │
│  • FP and vector units implicated                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2024 — OCP SERVER RESILIENCE                                               │
│  • Intel, AMD, NVIDIA, Arm join; SDC academic research awards               │
└─────────────────────────────────────────────────────────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2025 — IEEE, VERITAS                                                       │
│  • IEEE Computer: "The Dark Side of Computing" (Gizopoulos)                 │
│  • Veritas (HPCA): microarch-level SDC rate estimation                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

# Structure

```text
cpu_sdc_test/
├── CMakeLists.txt            # CMake project configuration             (Backend) (Config)
├── cpu_sdc_test.cpp          # SDC detection: CPU, GPU, RAM, SSD       (Backend) (Source)
├── gpu_sdc_cuda.cu           # NVIDIA CUDA GPU kernel                  (Backend) (Source)
├── gpu_sdc_cuda.h            # CUDA GPU test declarations              (Backend) (Source)
├── cpu_sdc_test_state.txt    # Persisted passed tests; skip on resume  (Backend) (Config)
└── README.md                 # Project documentation                   (Config)  (Docs)
```
