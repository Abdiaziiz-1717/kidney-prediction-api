from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from typing import List, Optional, Dict
import gc

app = FastAPI(
    title="Kidney Disease Prediction API",
    description="API for predicting kidney conditions from medical images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class PredictionRequest(BaseModel):
    image: str  # base64 encoded image
    models: List[str]  # list of model names to use

# Define model prediction result
class ModelPrediction(BaseModel):
    prediction_class: str
    confidence: float

# Define response model
class PredictionResponse(BaseModel):
    best_prediction: ModelPrediction
    all_predictions: Dict[str, ModelPrediction]
    description: Optional[str] = None

# Define the model architecture
class KidneyModel(nn.Module):
    def __init__(self, num_classes=3):
        super(KidneyModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Model paths
MODEL_PATHS = {
    'model1': 'models/best_kidney_model(95_accuracy).pth',
    'model2': 'models/best_kidney_model(92_accuracy).pth'
}

# Initialize models dictionary (will be populated on first use)
models = {}

def load_model(model_name: str) -> nn.Module:
    """Load a model if it's not already loaded"""
    if model_name not in models:
        if model_name not in MODEL_PATHS:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = MODEL_PATHS[model_name]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model instance
        model = KidneyModel()
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        models[model_name] = model
    
    return models[model_name]

def preprocess_image(image_data: str) -> torch.Tensor:
    """Preprocess the image for model input"""
    # Decode base64 image
    image_data = image_data.split(',')[1] if ',' in image_data else image_data
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    
    # Convert to numpy array
    image = np.array(image)
    
    # Convert to RGB if needed
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Define preprocessing transforms
    transform = A.Compose([
        A.Resize(224, 224),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Apply transforms
    transformed = transform(image=image)
    return transformed['image'].unsqueeze(0)  # Add batch dimension

def make_prediction(model: nn.Module, image_tensor: torch.Tensor) -> tuple:
    """Make prediction using the model"""
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class_index].item() * 100
    
    class_mapping = {0: 'NORMAL', 1: 'STONE', 2: 'TUMOR'}
    predicted_label = class_mapping.get(predicted_class_index, 'UNKNOWN')
    
    return predicted_label, confidence

@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "name": "Kidney Disease Prediction API",
        "version": "1.0.0",
        "description": "API for predicting kidney conditions from medical images",
        "endpoints": {
            "/predict": {
                "method": "POST",
                "description": "Predict kidney condition from an image",
                "input": {
                    "image": "base64 encoded image string",
                    "models": ["model1", "model2"]
                },
                "output": {
                    "best_prediction": {
                        "prediction_class": "NORMAL|STONE|TUMOR",
                        "confidence": "float between 0 and 100"
                    },
                    "all_predictions": {
                        "model1": {
                            "prediction_class": "NORMAL|STONE|TUMOR",
                            "confidence": "float between 0 and 100"
                        },
                        "model2": {
                            "prediction_class": "NORMAL|STONE|TUMOR",
                            "confidence": "float between 0 and 100"
                        }
                    },
                    "description": "Human-readable description of the prediction"
                }
            }
        },
        "available_models": list(MODEL_PATHS.keys())
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Preprocess image
        image_tensor = preprocess_image(request.image)
        
        # Get predictions from all requested models
        all_predictions = {}
        for model_name in request.models:
            try:
                model = load_model(model_name)
                predicted_class, confidence = make_prediction(model, image_tensor)
                all_predictions[model_name] = ModelPrediction(
                    prediction_class=predicted_class,
                    confidence=confidence
                )
            except Exception as e:
                print(f"Error with model {model_name}: {str(e)}")
                continue
        
        if not all_predictions:
            raise HTTPException(status_code=400, detail="No valid models specified")
        
        # Select prediction with highest confidence
        best_model = max(all_predictions.items(), key=lambda x: x[1].confidence)
        best_prediction = best_model[1]
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return PredictionResponse(
            best_prediction=best_prediction,
            all_predictions=all_predictions,
            description=f"The analysis indicates the presence of {best_prediction.prediction_class.lower()}."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 