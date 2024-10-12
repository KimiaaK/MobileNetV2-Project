import torch
from models.mobilenetv2 import MobileNetV2
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def evaluate(data_dir, model_path, num_classes=2, batch_size=16):
    # Define the data transformations (resize and convert to tensor)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Load the dataset for evaluation
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load the trained MobileNetV2 model
    model = MobileNetV2(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    # Initialize lists to store true labels and predictions
    all_labels = []
    all_preds = []

    # Evaluation loop
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Print the classification report
    report = classification_report(all_labels, all_preds, target_names=dataset.classes)
    print(report)

    # Calculate and print accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Accuracy: {accuracy:.4f}')

    # Plot the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, dataset.classes, rotation=45)
    plt.yticks(tick_marks, dataset.classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to dataset for evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    args = parser.parse_args()

    evaluate(data_dir=args.data, model_path=args.model)

