# AI Image Detection

A powerful AI-powered tool that helps distinguish between AI-generated images and real photographs using the EfficientNetB0 deep learning model.

## Features

- 🖼 Image classification for AI-generated vs real photographs
- ⚡ Fast and efficient processing

## How It Works

The system uses a sophisticated deep learning approach:

1. **Image Preprocessing**: Uploaded images are resized to 224×224 pixels and normalized
2. **Feature Analysis**: The model analyzes texture, patterns, and inconsistencies
3. **Classification**: Outputs a probability score to determine if the image is AI-generated or real

## Technical Details

- **Model**: EfficientNetB0
- **Input Size**: 224×224 pixels
- **Training Data**:
  - AI-generated images (from Midjourney, Stable Diffusion, DALL·E)
  - Real photographs (from open-image datasets and stock photography)
- **Data Augmentation**: Rotation, flipping, zooming, and color shifts
- **Training Split**: 80% training, 20% validation


## Getting Started

1. Clone the repository
2. Install dependencies
3. Run the application
4. Upload an image to test

## Accuracy

The model achieves less accuract becuase it was only trained on 5 epochs. If you have a powerful GPU then train this model at a higher epoch rate to increase the accuracy. 

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
