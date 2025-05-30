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
from typing import List, Optional

app = FastAPI()

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

# Define response model
class PredictionResponse(BaseModel):
    prediction_class: str
    confidence: float
    description: Optional[str] = None

# Model paths
MODEL_PATHS = {
    'model1': 'models/best_kidney_model(95_accuracy).pth',
    'model2': 'models/best_kidney_model(92_accuracy).pth'
}

# Initialize models
models = {}
for model_name, model_path in MODEL_PATHS.items():
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        models[model_name] = model

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

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Preprocess image
        image_tensor = preprocess_image(request.image)
        
        # Get predictions from all requested models
        predictions = []
        for model_name in request.models:
            if model_name in models:
                model = models[model_name]
                predicted_class, confidence = make_prediction(model, image_tensor)
                predictions.append({
                    'prediction_class': predicted_class,
                    'confidence': confidence
                })
        
        if not predictions:
            raise HTTPException(status_code=400, detail="No valid models specified")
        
        # Select prediction with highest confidence
        best_prediction = max(predictions, key=lambda x: x['confidence'])
        
        return PredictionResponse(
            prediction_class=best_prediction['prediction_class'],
            confidence=best_prediction['confidence'],
            description=f"The analysis indicates the presence of {best_prediction['prediction_class'].lower()}."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 