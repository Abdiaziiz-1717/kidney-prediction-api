# Kidney Prediction API

This API provides kidney disease prediction services using deep learning models. It can classify kidney images into three categories: NORMAL, STONE, and TUMOR.

## Features

- FastAPI-based REST API
- Support for multiple model predictions
- Image preprocessing with albumentations
- CORS enabled for cross-origin requests
- Base64 image input support

## API Endpoints

### POST /predict
Predicts kidney condition from an image.

Request body:
```json
{
    "image": "base64_encoded_image_string",
    "models": ["model1", "model2"]
}
```

Response:
```json
{
    "prediction_class": "NORMAL|STONE|TUMOR",
    "confidence": 95.5,
    "description": "The analysis indicates the presence of normal."
}
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Model Files

Place your trained model files in the `models` directory:
- `models/best_kidney_model(95_accuracy).pth`
- `models/best_kidney_model(92_accuracy).pth`

## Development

The API is built with:
- FastAPI
- PyTorch
- Albumentations
- OpenCV
- Pillow 