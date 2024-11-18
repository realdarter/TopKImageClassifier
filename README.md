
# TopKImageClassifier
The TopKImageClassifier returns the top K predicted categories and probabilities for input images.

# Requirements
- Python 3.0 or better
- Packages: torch, torchvision, pillow
   - You can install these using:  
     ```bash
     pip install torch torchvision pillow
     ```
   - Preferably install torch using NVIDIA GPU for accelerated training.  
     [Get GPU-enabled PyTorch](https://pytorch.org/get-started/locally/)

# Setup
- Open Terminal and navigate to your desired directory:  
  `{cd path_to_your_directory}`
- Clone the repository using:  
  ```bash
  git clone https://github.com/realdarter/TopKImageClassifier
  ```
- Run the main script to get started with the classifier.

# Example Dataset Structure
Ensure your dataset is structured as follows:
```
dataset/
├── garbage_classification/
│   ├── cardboard/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── metal/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── plastic/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
```
Each categorical folder (e.g., `cardboard`, `metal`, `plastic`) should contain images for classification.

# Features
- **Top-K Prediction**: The classifier can return the top K predicted categories along with their associated probabilities.
- **Easy Customization**: You can easily adjust the number of top predictions (K) and modify other parameters for your needs.

# Steps

1. **Install Dependencies**:  
   Make sure to install the required packages:  
   ```bash
   pip install torch torchvision pillow
   ```

2. **Prepare Dataset**:  
   Ensure your dataset is correctly structured as shown above. Each category should be in a separate subdirectory under `garbage_classification`.

3. **Train the Model**:  
   After setting up your dataset, you can start training the model by running the following code. Adjust training parameters as needed.

   ```python
   from image_classifier import *  # Ensure this imports the necessary functions and classes

   data_dir = 'data/garbage_classification'
   model_dir = 'saved_models/save1'
   
   args = create_args(
           num_epochs=2, 
           batch_size=32, 
           learning_rate=1e-5, 
           save_every=5, 
           )
   
   history = train(data_dir=data_dir, model_dir=model_dir, args=args)
   print(history)
   ```

4. **Test the Model**:  
   After training, you can test the model by inputting images and getting the top K predictions.

   ```python
   from image_classifier import *
   model_dir = 'saved_models/save1'

   args = create_args(
           top_k=5, 
           top_p=0.9, 
           )

   model, epoch, classes = load_model(model_dir)
   test_image_path = r'image path'
   top_k_labels, top_k_probs = predict_images(img_path=test_image_path, model=model, classes=classes, args=args)
   print(top_k_labels, top_k_probs)
   ```

# Notes
- You can fine-tune the model to your specific dataset by modifying the `train_model` function and adjusting training parameters such as `num_epochs`, `batch_size`, and `learning_rate`.
- The model will be saved to the specified directory for future use and can be loaded later for inference.

---
