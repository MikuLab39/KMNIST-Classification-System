# K49 Hiragana Classification System

## Project Overview
This project focuses on building a machine learning model to classify Japanese Hiragana characters using the **Kuzushiji-49 (K49)** dataset. The system covers the complete pipeline from data loading and Exploratory Data Analysis (EDA) to model training and evaluation. 

The goal is to demonstrate a reproducible machine learning workflow, utilizing Docker to ensure a consistent execution environment.

## Live Demo 
The system is deployed and available for live testing at the following address:
**Demo URL:** [https://k49.mikulab.com](https://k49.mikulab.com)

* **Web UI:** Access the link above to use the graphical interface for character prediction.
* **API Documentation:** Visit [https://k49.mikulab.com/docs](https://k49.mikulab.com/docs) to explore and test the interactive Swagger API reference.

## Installation & Requirements
To ensure the program runs correctly and is easy to reproduce, we strongly recommend using **Docker**. You can either build the image locally or pull the pre-built image.

**Prerequisites:**
* Docker installed on your machine.

### Option 1: Build from Source (Local Build)
If you want to verify the code and build the environment yourself:

1.  Clone the repository and add your model:
    ```bash
    git clone https://github.com/MikuLab39/K49-Classification-System.git
    ```
```bash
K49-Classification-System/
├── static/
├── src/
├── model.pth
├── Dockerfile
└── requirements.txt

 ```
2.  Build the Docker image using the provided `Dockerfile`:
    ```bash
    docker-compose up --build
    ```
3.  **Access the Web UI:**
    Once the container is running, open your browser and navigate to:
    `http://localhost:8339` (or the port defined in your configuration).

    > **Production Tip:** For a production environment, it is highly recommended to configure an **Nginx** reverse proxy and enable **HTTPS** for security. 

### Option 2: Quick Start with Docker Hub (Recommended)

1.  Ensure you have the `docker-compose.yml` file in your directory.
```bash
version: '3.8'

services:
  k49api-server:
    image: mikulab/k49-api:latest
    restart: always
    container_name: k49api-server
    ports:
      - "8339:8000"
    environment:
      - REDIS_URL=redis://k49api-redis:6379
    depends_on:
      - k49api-redis
    command: uvicorn src.api:app --host 0.0.0.0 --port 8000

  k49api-redis:
    image: redis:alpine
    container_name: k49api-redis

  k49api-worker:
    image: mikulab/k49-api:latest
    container_name: k49api-worker
    restart: always
    environment:
      - REDIS_URL=redis://k49api-redis:6379
    depends_on:
      - k49api-redis
    command: python -m src.worker

```

2.  Run the service in the background:
    ```bash
    docker-compose up -d
    ```
3.  **Access the Web UI:**
    Once the container is running, open your browser and navigate to:
    `http://localhost:8339` (or the port defined in your configuration).

    > **Production Tip:** For a production environment, it is highly recommended to configure an **Nginx** reverse proxy and enable **HTTPS** for security. 

# Web API Reference 

The system provides a RESTful API built with **FastAPI**. It supports both **synchronous** (real-time) and **asynchronous** (batch processing) predictions, designed to meet the advanced requirements for scalability.

> **Interactive Documentation:**
> Once the server is running, you can access the interactive Swagger UI at: `http://localhost:8339/docs`

## 1. General Info

### Health Check
Check if the API server is running correctly.

* **URL:** `/`
* **Method:** `GET`
* **Response:**
  ```json
  {
    "status": "ok",
    "version": "2.0"
  }

## 2. Synchronous Prediction (Single Image)

Use this endpoint for real-time inference of a single image. The server processes the request immediately and returns the result. This is suitable for low-latency requirements.

- **URL:** `/predict`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`

### Request Parameters

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` | Binary (File) | Yes | The image file to classify (jpg, png, etc.). |

### Example Request (cURL)

```bash
curl -X POST "http://localhost:8339/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./test_image.png"
  ```
### Example Response
```json
{
  "prediction": "あ",
  "class_id": 0,
  "confidence": 0.98
}
```
## 3. Asynchronous Batch Prediction (Multiple Images)

Use this endpoint for processing multiple images at once. This is designed for scalability and high-throughput scenarios. The API returns a `task_id` immediately, while the prediction process runs in the background.

- **URL:** `/batch_predict`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`

### Request Parameters

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `files` | Array of Binary | Yes | A list of image files to classify. |

### Example Request (cURL)

```bash
curl -X POST "http://localhost:8339/batch_predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@./image1.png" \
  -F "files=@./image2.png"
```
### Example Response
Returns a Task ID for tracking the job status.
```json
{
  "task_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "processing"
}
```
## 4. Get Task Result

Retrieve the results of an asynchronous batch prediction using the task_id.

- **URL:** `/tasks/{task_id}`
- **Method:** `GET`

### Path Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `task_id` | String | Yes | The UUID received from the `/batch_predict` endpoint. |

### Example Request (cURL)

**Bash**
```bash
curl -X GET "http://localhost:8339/tasks/3fa85f64-5717-4562-b3fc-2c963f66afa6"
```
### Example Response
```json
{
  "task_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "completed",
  "results": [
    {
      "filename": "image1.png",
      "prediction": "あ",
      "confidence": 0.98
    },
    {
      "filename": "image2.png",
      "prediction": "い",
      "confidence": 0.95
    }
  ]
}
```

## Dataset Preparation
This project utilizes the **Kuzushiji-49 (K49)** dataset.
Please download the dataset from the official link: [KMNIST Dataset](http://codh.rois.ac.jp/kmnist/index.html.en)

1. Download the `.npz` files for Kuzushiji-49.
2. Place them in the `data/k49/` directory so the structure looks like this:
   ```text
   K49-Classification-System/
   ├── data/
   │   └── k49/
   │       ├── k49-train-imgs.npz
   │       ├── k49-train-labels.npz
   │       ├── k49-test-imgs.npz
   │       └── k49-test-labels.npz


## Model Development (Jupyter Notebook)

For a detailed walkthrough of the machine learning pipeline—including **Exploratory Data Analysis (EDA)**, **Model Training**, and **Performance Verification**—please refer to the provided Jupyter Notebook.

This notebook demonstrates the complete workflow for the project, covering data distribution analysis, model architecture selection, and evaluation metrics.

