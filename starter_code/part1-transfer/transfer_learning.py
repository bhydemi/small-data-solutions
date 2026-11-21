"""Train a three-class travel image classifier with a frozen ImageNet VGG16."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from TrainModel import train_model
from TestModel import test_model


DATA_DIR = Path(__file__).resolve().parent / "imagedata-50"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 10
NUM_WORKERS = 0  # Portable on Windows, macOS notebooks, and constrained environments.


def build_loaders():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_data = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_data = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transform)
    test_data = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)
    # Keep class indices consistent across all splits.
    assert train_data.class_to_idx == val_data.class_to_idx == test_data.class_to_idx
    common = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    return (
        torch.utils.data.DataLoader(train_data, shuffle=True, **common),
        torch.utils.data.DataLoader(val_data, shuffle=False, **common),
        torch.utils.data.DataLoader(test_data, shuffle=False, **common),
        train_data.classes,
    )


def build_model(num_classes):
    # Newer torchvision uses weights; older versions use pretrained.
    try:
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    except (AttributeError, TypeError):
        model = models.vgg16(pretrained=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    # VGG16's final classifier layer maps 4096 features to ImageNet's 1000 classes.
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    for parameter in model.classifier[6].parameters():
        parameter.requires_grad = True
    return model


def main():
    train_loader, val_loader, test_loader, class_names = build_loaders()
    model = build_model(len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[6].parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trained_model = train_model(
        model, criterion, optimizer, scheduler,
        train_loader, val_loader, num_epochs=8,
    )
    test_model(test_loader, trained_model, class_names)
    # The ImageNet backbone is reproducibly loaded by build_model; save only the
    # small task-specific head instead of duplicating the 500+ MB VGG16 weights.
    torch.save({"classifier_state_dict": trained_model.classifier[6].state_dict(),
                "class_names": class_names},
               DATA_DIR.parent / "travel_vgg16.pth")


if __name__ == "__main__":
    main()
