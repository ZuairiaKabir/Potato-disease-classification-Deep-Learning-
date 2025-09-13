
# Potato Disease Detection API

[![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.101.0-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow)](https://www.tensorflow.org/)

An API for detecting potato diseases (Early Blight, Late Blight, Healthy) from images using a TensorFlow model. The API is built with **FastAPI** and tested locally using **Postman**.

---

## Features

- Upload an image of a potato leaf and get predictions:
  - **Class**: Early Blight / Late Blight / Healthy
  - **Confidence score**
- Simple FastAPI server with CORS enabled for local testing
- Uses TensorFlow SavedModel loaded as a `TFSMLayer`

---

## Getting Started

### Prerequisites

- Python 3.10+
- TensorFlow 2.20
- FastAPI
- Uvicorn
- Pillow
- Numpy

Install dependencies:

```bash
pip install -r requirements.txt
````

> Note: The dataset is **not included**. It was downloaded from [Kaggle - PlantVillage Potato Dataset](https://www.kaggle.com/datasets) as shown in the tutorial.

---

### Running the API

1. Navigate to the API folder:

```bash
cd python/potato_disease/api
```

2. Run the server:

```bash
uvicorn main:app --reload --host localhost --port 8000
```

3. Open [Postman](https://www.postman.com/) and test endpoints:

* **Ping endpoint:**
  `GET http://localhost:8000/ping` → returns `"Hello, I am alive"`

* **Prediction endpoint:**
  `POST http://localhost:8000/predict`
  Form-data: `file` → Upload an image of a potato leaf

---

### Project Structure

```
potato_disease/
│
├── api/
│   ├── main.py          # FastAPI server
│   └── requirements.txt # Dependencies
│
├── saved_models/        # TensorFlow SavedModel versions
│   └── 1/
│
└── README.md
```

---

### References

* [YouTube Tutorial Playlist](https://www.youtube.com/playlist?list=PLeo1K3hjS3utJFNGyBpIvjWgSDY0eOE8S)
* [PlantVillage Potato Dataset on Kaggle](https://www.kaggle.com/datasets)

---

### Notes

* Model input images are **256x256** RGB and normalized (0-1).
* The API has only been tested locally with Postman; no deployment yet.

