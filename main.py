import torch
from models.mobilenetv2 import MobileNetV2
from PIL import Image
import torchvision.transforms as transforms

def load_model(model_path='models/mobilenetv2.pth', num_classes=2):
    # Load the MobileNetV2 model with the correct number of classes
    model = MobileNetV2(num_classes=num_classes)
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

def predict(model, image_path):
    # Define the transformations for the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Load and preprocess the image
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make the prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

if __name__ == "__main__":
    import argparse
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to the image for prediction')
    args = parser.parse_args()

    # Load the model
    model = load_model()
    print("Model loaded successfully.")

    # Make the prediction
    prediction = predict(model, args.image)
    print(f"Predicted Class: {prediction}")
