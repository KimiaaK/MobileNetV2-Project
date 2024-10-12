# MobileNetV2-Project
# MobileNetV2 Strawberry Disease Detection

This project applies MobileNetV2 for detecting strawberry diseases using a deep learning model. The original code has been modularized for professional deployment, including Dockerization.

## Project Structure

- `models/` contains the MobileNetV2 model code.
- `scripts/` contains training, evaluation, and utility scripts.
- `notebooks/` contains the original Jupyter notebook.
- `main.py` runs the primary functionality for making predictions.
- `Dockerfile` and `.dockerignore` are used for containerization.

## Getting Started

### Prerequisites

- Docker
- Python 3.9

### Setup

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/MobileNetV2-Project.git
   cd MobileNetV2-Project
   ```

2. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. Build and run the Docker container:

   ```sh
   docker build -t mobilenetv2-project .
   docker run -p 8080:8080 mobilenetv2-project
   ```

## Usage

### Training the Model

To train the model on your dataset, use the following command:

```sh
python scripts/train.py --data data/dataset_path --epochs 10 --batch_size 16 --learning_rate 0.001
```

- `--data`: Path to the training dataset.
- `--epochs`: Number of training epochs (default is 10).
- `--batch_size`: Batch size for training (default is 16).
- `--learning_rate`: Learning rate for the optimizer (default is 0.001).

### Evaluating the Model

To evaluate the trained model on a test dataset, use the following command:

```sh
python scripts/evaluate.py --data data/test_dataset --model models/mobilenetv2.pth
```

- `--data`: Path to the evaluation dataset.
- `--model`: Path to the trained model file (`mobilenetv2.pth`).

This script will print a classification report, display a confusion matrix, and calculate accuracy to help assess the model's performance.

### Making Predictions

To use the trained model to make predictions on a new image:

```sh
python main.py --image path/to/sample_image.jpg
```

- `--image`: Path to the image for which you want to make a prediction.

## Project Components

### Dockerfile

The `Dockerfile` is used to create a Docker container for the project. It sets up the Python environment, installs dependencies, and runs `main.py`.

### .dockerignore and .gitignore

- **.dockerignore**: Specifies which files and directories should be ignored when building the Docker image (e.g., cache files, notebook checkpoints).
- **.gitignore**: Specifies which files should be ignored by Git (e.g., compiled Python files, environment directories).

## Evaluation Metrics

The evaluation script (`evaluate.py`) provides the following metrics:

- **Accuracy**: The ratio of correctly predicted samples to the total samples.
- **Classification Report**: Precision, recall, F1-score, and support for each class.
- **Confusion Matrix**: A visualization of how well the model is classifying each class, helping identify misclassifications.

## License

This project is licensed under the MIT License.
```
    }
  ]
}
![image](https://github.com/user-attachments/assets/5e29dea8-3617-4e6d-8546-a7da4ba6527d)
