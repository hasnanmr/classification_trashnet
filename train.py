from datasets import load_dataset
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
import wandb
import os


# Get the W&B API key from the environment variable
wandb_api_key = os.getenv('wan_db')
if wandb_api_key:
    wandb.login(key=wandb_api_key, relogin=True)  
else:
    raise ValueError("W&B API key not found in environment variables")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Custom Dataset class
class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Assuming the dataset returns a dictionary with 'image' and 'label' keys
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']
        
        # The image is already a PIL Image object, so no need to convert it
        if self.transform:
            image = self.transform(image)
        return image, label

# Load dataset
dataset = load_dataset("garythung/trashnet")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Create an instance of the custom dataset
full_dataset = TransformDataset(dataset['train'], transform=transform)

# Split the dataset into train, validation, and test sets
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Print sizes of each split
print(f'Train dataset size after split: {len(train_dataset)}')
print(f'Validation dataset size after split: {len(val_dataset)}')
print(f'Test dataset size after split: {len(test_dataset)}')

# Create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def get_model():
    "A simple model"
    model_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in model_resnet.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_ftrs, 6)  

    model_resnet = model_resnet.to(device)
    return model_resnet

def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    num_classes = 6
    table = wandb.Table(columns=["image", "pred", "target"] + [f"score_{i}" for i in range(num_classes)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img.numpy().transpose(1, 2, 0)), pred.item(), targ.item(), *prob.numpy())
    wandb.log({"predictions_table": table}, commit=False)

def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    model.eval()
    val_loss = 0.
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(device), labels.to(device)

            # Forward pass ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels)*labels.size(0)

            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Log one batch of images to the dashboard, always same batch_idx.
            if i==batch_idx and log_images:
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)

import math
# Launch 5 experiments, trying different dropout rates
for _ in range(15):
    # 🐝 initialise a wandb run
    wandb.init(
        project="classification_trashnet",
        config={
            "epochs": 15,
            "batch_size": 32,
            "lr": 1e-4,
            })
    
    # Copy your config 
    config = wandb.config

    # Get the data
    train_dl = train_loader
    valid_dl = val_loader
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)
    
    # A simple MLP model
    model = get_model()

    wandb.watch(model, log="all")

    # Make the loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

   # Training
    example_ct = 0
    step_ct = 0
    for epoch in range(config.epochs):
        model.train()
        for step, (images, labels) in enumerate(train_dl):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            train_loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            example_ct += len(images)
            metrics = {"train/train_loss": train_loss, 
                       "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch, 
                       "train/example_ct": example_ct}
            
            if step + 1 < n_steps_per_epoch:
                # 🐝 Log train metrics to wandb 
                wandb.log(metrics)
                
            step_ct += 1

        val_loss, accuracy = validate_model(model, valid_dl, loss_func, log_images=(epoch==(config.epochs-1)))

        # 🐝 Log train and validation metrics to wandb
        val_metrics = {"val/val_loss": val_loss, 
                       "val/val_accuracy": accuracy}
        wandb.log({**metrics, **val_metrics})
        
        print(f"Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}")

    # # If you had a test set, this is how you could log it as a Summary metric
    # wandb.summary['test_accuracy'] = 0.8

    # 🐝 Close your wandb run 
    wandb.finish()
