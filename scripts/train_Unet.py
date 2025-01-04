from imports import *
from model import UNet
from dataLoader import get_data_loaders
from utils import train_model

# Define paths to your folders
A_folder = '/Users/ehzalp/Desktop/ChangeDetection/data/preprocessedData/images_A'
B_folder = '/Users/ehzalp/Desktop/ChangeDetection/data/preprocessedData/images_B'
labels_folder = '/Users/ehzalp/Desktop/ChangeDetection/data/preprocessedData/masks'

# Define batch size and other parameters
batch_size = 16
shuffle = True
num_workers = 2
num_epochs = 50

# Get data loaders
dataloaders = get_data_loaders(A_folder, B_folder, labels_folder, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# Define model
model = UNet(n_classes=1)  # Define number of classes appropriately

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define scheduler (optional)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Function to train the model
def main():
    # Train the model
    trained_model = train_model(model, optimizer, scheduler, dataloaders, num_epochs=num_epochs)

    # Save best model
    torch.save(trained_model.state_dict(), 'best_model_unet_50ep.pth')

if __name__ == "__main__":
    main()  # Only run this when the script is executed directly