# House Problem Images Classification

This project fine-tunes a Vision Transformer (ViT) model to classify images related to house problems into four categories: **builder**, **electrician**, **others**, and **plumber**. The model is trained on the `MadanKhatri/house_problem_images` dataset from Hugging Face and achieves high accuracy for image classification tasks.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Inference](#inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to classify images of house-related problems into one of four categories using a fine-tuned Vision Transformer (ViT) model. The model is based on the `google/vit-base-patch16-224-in21k` pre-trained model and is fine-tuned on a custom dataset. The project includes data preprocessing, model training, evaluation, and inference scripts.

## Dataset
The dataset used is `MadanKhatri/house_problem_images`, available on Hugging Face. It contains images labeled into four categories:
- **builder**
- **electrician**
- **others**
- **plumber**

### Dataset Structure
- **Train**: 1,758 images
- **Test**: 311 images
- **Features**: Each sample includes an `image` (PIL Image) and a `label` (integer corresponding to the class).

## Installation
To run this project, you need to install the required dependencies. Follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/house-problem-classification.git
   cd house-problem-classification
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Ensure you have Python 3.10 or later installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. **Log in to Hugging Face**:
   To access the dataset and push the model to the Hugging Face Hub, log in using:
   ```bash
   huggingface-cli login
   ```
   Alternatively, run `notebook_login()` in a Jupyter notebook as shown in the code.

## Usage
The project includes scripts for data preprocessing, model training, evaluation, and inference. Below are the steps to use the code.

### 1. Preprocessing
The dataset is preprocessed using the `ViTFeatureExtractor` from the `transformers` library. Images are transformed with:
- **Training**: Random resized crop, random horizontal flip, normalization.
- **Validation**: Resize, center crop, normalization.

### 2. Training
The model is fine-tuned using the `Trainer` API from Hugging Face. To train the model, run the provided script or notebook. Key training parameters:
- **Model**: `google/vit-base-patch16-224-in21k`
- **Epochs**: 10
- **Batch Size**: 16
- **Learning Rate**: 2e-4
- **Output Directory**: `finetuned-occupations`
- **Push to Hub**: Enabled (model is saved to `MadanKhatri/finetuned-occupations`)

### 3. Inference
To classify a new image, use the following steps:
1. Load the fine-tuned model and processor:
   ```python
   from transformers import AutoImageProcessor, AutoModelForImageClassification
   processor = AutoImageProcessor.from_pretrained("MadanKhatri/finetuned-occupations")
   model = AutoModelForImageClassification.from_pretrained("MadanKhatri/finetuned-occupations")
   ```
2. Prepare and classify an image:
   ```python
   from PIL import Image
   import torch

   image = Image.open("path/to/your/image.png").convert("RGB")
   encoding = processor(image, return_tensors="pt")
   with torch.no_grad():
       outputs = model(**encoding)
       logits = outputs.logits
   predicted_class_idx = logits.argmax(-1).item()
   print("Predicted class:", model.config.id2label[predicted_class_idx])
   ```

## Training the Model
The model is trained using the `Trainer` class from the `transformers` library. The training script:
- Loads the dataset and applies preprocessing.
- Configures training arguments (e.g., batch size, epochs, learning rate).
- Trains the model and evaluates it on the test set.
- Saves the model and metrics to the output directory and pushes to the Hugging Face Hub.

To train the model, run the notebook or extract the relevant code into a Python script. Ensure you have a GPU for faster training (FP16 is enabled).

## Results
The model achieves the following performance on the test set after 10 epochs:
- **Accuracy**: 91.96%
- **Validation Loss**: 0.3327

Training metrics are logged to TensorBoard, and the best model is saved based on accuracy.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure your code follows the project's coding standards and includes appropriate tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
