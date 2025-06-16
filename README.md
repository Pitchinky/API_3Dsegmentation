# API 3D Segmentation

This project provides an API for performing segmentation on textured 3D objects.

## 🧠 Overview

The API processes 3D models (such as `.obj` files with textures) and uses computer vision and machine learning tools to identify and segment wound areas or regions of interest. It leverages both geometric and image-based data to produce accurate segmentations.

## ⚙️ Technologies Used

- **Python 3.10**
- **FastAPI** — for building the RESTful API
- **OpenCV, NumPy, scikit-learn** — for image processing and clustering
- **Open3D** — for 3D mesh rendering and manipulation
- **Deepskin** — external library for wound segmentation (imported via GitHub)

## 🚀 Getting Started

### 1. Install Dependencies

Make sure you have Python 3.10 installed, then run:

```bash
pip install -r requirements.txt
