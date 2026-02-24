# PlantVillage Disease Detection Wiki

Welcome to the comprehensive documentation brain for the PlantVillage Disease Detection project. This wiki serves as the central hub for understanding the system architecture, codebase, data workflows, and current challenges.

## 🌟 Project Overview

This project implements a complete, open-source pipeline for training and deploying plant disease detection models on Apple Silicon Macs. It focuses on privacy-preserving, edge-first agricultural diagnostics using the state-of-the-art YOLO26n architecture. Key innovations include native Apple Silicon (MPS) optimization, robust handling of GPU bottlenecks, and INT8 CoreML quantization for mobile deployment.

## 🚀 Quick Start
To get started with the project quickly, refer to the [Main Repository README](../../README.md) or the [QUICKSTART.md](../QUICKSTART.md).

For setting up the environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/leafmd/cli.py check
```

## 📊 System State

- **Current Status**: Production-ready pipeline for Apple Silicon (M4 Max optimized).
- **Core Technologies**: PyTorch 2.6+, Ultralytics (YOLO26n), CoreMLTools, Python 3.11+.
- **Active Challenges**: Managing Apple Silicon MPS memory constraints and coordinate corruption bugs.

## 📚 Table of Contents

1. [Architecture Overview](01_architecture.md)
2. [Code Reference](02_code_reference.md)
3. [CLI / API Reference](03_api_reference.md)
4. [Configuration Guide](04_configuration.md)
5. [Data Workflow](05_data_workflow.md)
6. [Current Challenges & Technical Debt](06_current_challenges.md)
7. [Troubleshooting Guide](07_troubleshooting.md)
8. [Project Journey & Evolution](08_project_journey.md)

### Additional Resources
- [Checking Results](checking_results.md)
- [Interpreting Plots](interpreting_plots.md)
