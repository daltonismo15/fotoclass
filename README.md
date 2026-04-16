# Blur Classifier

A deep learning model that automatically classifies photos into three categories:
- **Sharp** — correctly focused photos
- **Defocused blurred** — photos with incorrect focus
- **Motion blurred** — photos affected by camera or subject movement

## 🚀 Try it out
[**Open the app on Hugging Face**](https://huggingface.co/spaces/Daltonismo15/fotoclass)

Upload your photos and download a zip file with the images already sorted into three folders.

## Model
- Architecture: ResNet18 (fine-tuned)
- Training dataset: [Blur Dataset by kwentar](https://www.kaggle.com/datasets/kwentar/blur-dataset)
- Training accuracy: ~94%
- Framework: fastai

## Project Structure
blur-classifier 
- app.py              # Gradio web app
- requirements.txt    # Dependencies
- notebook.ipynb      # Training notebook

## How it works
1. Upload a zip file with your jpeg photos in it
2. The model classifies each photo using a fine-tuned ResNet18
3. Download a zip file with three folders: sharp/, defocused_blurred/, motion_blurred/

## Future improvements
- Retrain on a custom dataset of real photography for better accuracy
- Migrate to PyTorch + torchvision for better compatibility
- Add confidence threshold to flag uncertain predictions
