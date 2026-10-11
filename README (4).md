# DGX Spark (GB10) Benchmark Studio

A Python-based benchmarking suite for NVIDIA DGX Spark systems powered by the GB10 Grace Blackwell Superchip. It provides command-line and Flask dashboard workflows for measuring GPU compute performance, memory bandwidth, host/device transfers, and sustained BF16 workload behavior.

> **Status:** Treat benchmark output as experimental until you validate the kernels, precision support, timing methodology, and reported throughput on your system.

## 3D DGX Spark Visualizer

Explore the DGX Spark hardware concept with the standalone interactive Three.js cutaway visualizer:

**[Open the 3D DGX Spark Visualizer](dgx-spark-visualizer.html)**

The visualizer includes camera presets for the SoC, GPU, CPU, memory, NVLink, and fan, plus shell/X-ray controls, rotation, and a simulated workload/thermal display. It is a **visual simulation**, not a live hardware monitor, and its displayed specifications or performance figures should be independently verified.

The HTML page loads Three.js from a CDN, so an internet connection may be required. GitHub does not execute JavaScript embedded in a README; open the linked HTML file in a browser or host it as a web page to use the interactive scene.

## Features

- System and CUDA environment information
- GEMM benchmarks across supported precisions, including FP32, TF32, FP16, and BF16
- Optional FP8 and FP4 experiments where the installed software stack and hardware kernels support them
- CPU-versus-GPU compute comparisons
- GPU memory bandwidth tests
- Host-to-device (H2D) and device-to-host (D2H) transfer measurements
- Sustained BF16 workload with telemetry sampling
- JSON output and optional PDF reports
- Flask-based dashboard (requires `dashboard.html`)

The exact tests available depend on the implementation, installed libraries, GPU capabilities, and command-line options.

## Requirements

- NVIDIA DGX Spark / GB10 system, or a compatible NVIDIA CUDA environment
- A supported NVIDIA driver and CUDA-enabled PyTorch build
- Python 3.10+ recommended
- Flask for dashboard mode
- ReportLab for PDF report generation
- The project's `dashboard.html` frontend for dashboard mode

FP8/FP4 paths may require specific PyTorch versions, libraries, kernels, or hardware support. A test that is unavailable or unsupported should be reported as such rather than treated as a valid measurement.

## Setup

1. Save the Python script as `spark_studio.py`.
2. Put `dashboard.html` in the same directory as `spark_studio.py` if you intend to use the web dashboard.
3. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, use:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

4. Install the dependencies used by the script:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install torch flask reportlab
   ```

   Install a PyTorch build appropriate for your platform and NVIDIA software stack. Follow the official PyTorch installation instructions if the generic command installs an unsuitable build.

5. Verify PyTorch can see CUDA:

   ```bash
   python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
   ```

   Do not run GPU benchmarks until CUDA is available and the expected device is detected.

## Running the benchmark suite

### Web dashboard

Start the application:

```bash
python spark_studio.py
```

Then open:

```text
http://127.0.0.1:5000
```

Keep the dashboard bound to localhost unless you have deliberately configured authentication and network access controls. Do not expose a development Flask server directly to an untrusted network.

### Command-line mode

Run the headless benchmark workflow:

```bash
python spark_studio.py --cli
```

Run an example with a matrix size of 8192 and a 60-second sustained workload:

```bash
python spark_studio.py --cli --size 8192 --sustained 60
```

The example assumes the script defines the `--size` and `--sustained` options. Check the script's `--help` output for the authoritative list of flags:

```bash
python spark_studio.py --help
```

Large matrix sizes and sustained workloads can use substantial memory, power, and thermal headroom. Start with smaller settings, monitor the system, and increase workloads gradually.

## Typical workflow

1. **Prepare the system:** close unrelated GPU-heavy workloads and confirm the device is healthy.
2. **Verify the environment:** check the driver, PyTorch version, CUDA availability, and device name.
3. **Run a baseline:** execute a short run to confirm all selected tests work.
4. **Measure compute:** run supported GEMM precision tests and record throughput.
5. **Measure memory and transfers:** run memory bandwidth and H2D/D2H tests.
6. **Measure sustained behavior:** enable a sustained BF16 workload and monitor throughput and telemetry.
7. **Review results:** inspect the JSON output and optional PDF report; compare only runs made under similar conditions.

## Reports and output

The script is designed to write reports into a `reports/` directory. Depending on the implementation and installed dependencies, outputs may include:

- **JSON:** structured system details and benchmark measurements, suitable for post-processing
- **PDF:** a human-readable summary with tables, charts, or highlights (requires ReportLab)

Check the script's output messages and report-writing functions for the exact filenames and schema. Keep original JSON reports when comparing runs so measurements can be audited.

## Benchmarking guidance

For more reproducible results:

- Run the same tests with the same matrix sizes and iteration counts.
- Warm up kernels before timing.
- Synchronize CUDA correctly around timed GPU work.
- Use repeated measurements and report a median or another clearly defined statistic.
- Separate allocation and compilation/setup time from steady-state kernel time when appropriate.
- Record power mode, clocks, temperature, driver, CUDA, and PyTorch versions.
- Avoid comparing results collected while other workloads are using the GPU.
- Make sure the FLOP count matches the actual operation and that units are calculated consistently.
- Treat CPU-versus-GPU results as comparisons of the tested implementations, not as universal hardware rankings.

### Precision and metric caveats

- **TF32** behavior depends on PyTorch settings and the operation used.
- **FP8/FP4** support is kernel- and software-stack-dependent. Confirm input formats, scaling, accumulation precision, layouts, and correctness before interpreting performance.
- **Memory bandwidth** calculations must use the correct bytes transferred and account for the measurement method.
- **H2D/D2H** results depend on transfer size, pinned memory, synchronization, and whether setup costs are included.
- **Sustained throughput** should be computed from measured work over the matching measurement interval. Counter resets and sampling windows must not distort the result.
- A high throughput number alone does not establish numerical correctness. Validate outputs against an appropriate reference and tolerance.

## Troubleshooting

### CUDA is unavailable

- Confirm the NVIDIA driver is installed and healthy.
- Check that the installed PyTorch build supports CUDA for the target platform.
- Run the verification command from the same Python environment used to launch the benchmark.

### Dashboard does not load

- Confirm Flask is installed.
- Confirm `dashboard.html` is present at the path expected by the Python script.
- Read the terminal output for the actual host, port, and error traceback.

### An FP8 or FP4 test fails

- Check the GPU, PyTorch, CUDA, and kernel/library support required by that test.
- Treat unsupported formats as unavailable; do not silently substitute another precision while retaining the original test label.

### Results vary between runs

- Check background GPU usage, temperature, clocks, power limits, and workload size.
- Use warmups and repeated measurements.
- Keep software versions and test parameters consistent.

### PDF report is missing

- Install ReportLab in the active environment.
- Check the terminal output and write permissions for the report directory.

## Safety and operational notes

- Benchmarking can draw significant power and create sustained thermal load. Monitor system telemetry and follow the platform's operating guidance.
- Do not expose the Flask development server to untrusted networks.
- Reports may contain system and hardware details; review them before sharing publicly.

## Project layout

A typical layout is:

```text
.
├── spark_studio.py
├── dashboard.html
└── reports/
    ├── ...json
    └── ...pdf
```

The `dashboard.html` and `reports/` entries depend on the files generated or supplied by your setup.

## License

No license information was included with the supplied script. Add a `LICENSE` file before distributing the project, and ensure any bundled dependencies or frontend assets comply with their respective licenses.
