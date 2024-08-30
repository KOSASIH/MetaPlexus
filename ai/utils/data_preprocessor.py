import torchvision.transforms as transforms
import numpy as np

def get_default_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_augmented_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image_file, transform):
    image = Image.open(image_file)
    image = transform(image)
    return image

def preprocess_images(image_files, transform):
    preprocessed_images = []
    for image_file in image_files:
        image = preprocess_image(image_file, transform)
        preprocessed_images.append(image)
    return preprocessed_images
