import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import sys

def get_model(num_classes):
    """Rebuilds the EfficientNet_V2_S architecture."""
    # Load base model
    model = models.efficientnet_v2_s()
    # Match the 62 classes from training
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def predict(image_path, model_path, classes_path):
    # Determine device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the Classes array
    if not os.path.exists(classes_path):
        print(f"Error: Classes file '{classes_path}' not found.")
        return
    
    # allow_pickle is True because the array contains strings
    classes = np.load(classes_path, allow_pickle=True)
    num_classes = len(classes)

    # 2. Load the Model
    model = get_model(num_classes)
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights '{model_path}' not found.")
        return

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Define Image Transformations (Must match training exactly)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3), # Essential for stroke detection
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. Load and Process Image
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # 5. Perform Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        
    # Get the character string using the index
    final_char = classes[predicted.item()]
    return final_char

if __name__ == "__main__":
    MODEL_PATH = '../models/final_handwriting_model.pth'
    CLASSES_PATH = '../classes.npy'
    
    # Use image from command line or default to test_image.png
    img_to_test = sys.argv[1] if len(sys.argv) > 1 else 'test_letters/k.png'

    print(f"Checking: {img_to_test}")
    
    result = predict(img_to_test, MODEL_PATH, CLASSES_PATH)
    
    if result:
        print(f"\n" + "="*25)
        print(f" PREDICTED: {result}")
        print("="*25)