# 🫁 Yasor Medical-Vision API (Chest X-ray Diagnosis)
---

[![Production Ready](https://img.shields.io/badge/Production--Ready-Yes-green.svg)](https://github.com/YassinSanad/Yasor-Medical-Vision)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg?style=flat&logo=docker)](https://www.docker.com/)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime-orange.svg)](https://onnxruntime.ai/)

A production-grade, high-performance, and privacy-focused Microservice for real-time Chest X-ray diagnostic inference. Built and optimized by Yasor Labs, this system leverages deep learning to detect respiratory conditions with maximum throughput and minimal resource footprint, designed specifically for On-Premises hospital deployments.

---

## 🚀 Key Engineering Features

* Advanced Deep Learning Backbone: Powered by a customized ResNet50 architecture fine-tuned with rigorous optimization strategies to handle medical imaging clinical imbalance.
* ONNX Execution Engine: The native PyTorch model is converted to ONNX (Open Neural Network Exchange) with full Graph Optimization (Layer Fusion & Constant Folding). 
* Lightweight Footprint: By migrating the inference from PyTorch to ONNX Runtime, the final production Docker image is stripped of heavy deep learning training frameworks, reducing the container size by over 2 GB and slashing RAM usage.
* Production-Grade Infrastructure: Fully containerized using Docker and served via asynchronous FastAPI endpoints with integrated health-checks, structured JSON logging, and automatic Swagger UI generation.

---

## 🗺️ System Architecture

1. PyTorch Training Artifact (.pth weights) $\rightarrow$ Optimized & Exported via torch.onnx.
2. ONNX Graph Optimization $\rightarrow$ Static layers fused and folded for maximum execution speed.
3. FastAPI Web Service $\rightarrow$ Exposes clean HTTP endpoints for image uploading and asynchronous batch-ready processing.
4. Docker Packaging $\rightarrow$ Encapsulated inside a hardened, isolated Linux environment ready for immediate deployment on edge nodes or cloud servers.

---

## 🛠️ Tech Stack & Requirements

* Language: Python 3.10
* API Framework: FastAPI & Uvicorn
* Inference Runtime: ONNX Runtime
* Image Processing: Pillow & NumPy
* DevOps & Containerization: Docker

---

# 🔒 Security & Privacy Compliance
​Hospital networks enforce strict regulations regarding patient data sovereignty. Yasor Medical-Vision API is explicitly engineered to run 100% locally (On-Premises). No patient data, diagnostic logs, or raw imagery ever leaves your dedicated hardware infrastructure, fulfilling institutional compliance guidelines and preserving total medical privacy.

# ​🏢 About Yasor Labs
​Yasor Labs is a cutting-edge AI engineering entity specializing in production-grade automated systems, high-performance Deep Learning pipelines, and zero-latency Embedded/Edge AI solutions. We bridge the gap between academic machine learning and resilient, scalable enterprise architectures
