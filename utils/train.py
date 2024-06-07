import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from losses import ContrastiveLoss
import os


def train_model(model, train_dataset, val_dataset, checkpoint_folder, num_epochs=10, batch_size=32,
                learning_rate=0.001):
    """
    Train the model using the provided datasets.

    Args:
    - model: The model to be trained
    - train_dataset: Dataset for training
    - val_dataset: Dataset for validation
    - checkpoint_folder: Folder to store checkpoints
    - num_epochs: Number of epochs for training
    - batch_size: Batch size for training
    - learning_rate: Learning rate for optimization

    Returns:
    - model: Trained model
    - train_losses: List of training losses
    - val_losses: List of validation losses
    """
    # Create the checkpoint folder if it doesn't exist
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # Define data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = ContrastiveLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Lists to store training and validation losses
    train_losses = []
    val_losses = []

    # Variables to keep track of the best model and its performance
    best_val_loss = float('inf')
    best_model_state = None

    model = model.to(device)
    print("Training started...")
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        print("*" * 100)
        print(f"Epoch [{epoch + 1}/{num_epochs}]:")
        model.train()
        running_train_loss = 0.0
        for i, (inputs, labels, _) in enumerate(train_loader):
            optimizer.zero_grad()
            # Forward pass
            inputA = inputs[0].to(device)
            inputB = inputs[1].to(device)
            labels = labels.to(device)
            output1, output2 = model(inputA, inputB)
            # Compute loss
            loss = criterion(output1, output2, labels)
            # Backward pass
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

            if i % 200 == 0:
                print(f"\t Batch [{i}/{len(train_loader)}], Train Loss: {loss.item():.4f}")

        # Compute average training loss for the epoch
        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for i, (inputs, labels, _) in enumerate(val_loader):
                inputA = inputs[0].to(device)
                inputB = inputs[1].to(device)
                labels = labels.to(device)
                output1, output2 = model(inputA, inputB)
                loss = criterion(output1, output2, labels)
                running_val_loss += loss.item()

                if i % 100 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], Validation Batch [{i}/{len(val_loader)}], Val Loss: {loss.item():.4f}")

        # Compute average validation loss for the epoch
        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        # Save the model checkpoint for every epoch (last model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': epoch_val_loss
        }, os.path.join(checkpoint_folder, f'last.pt'))

        # Save the best model checkpoint based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }, os.path.join(checkpoint_folder, f'best.pt'))

        # Print progress
        print(f"Validation, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        print("*" * 100)
        scheduler.step()
    print("Training completed.")

    return model, train_losses, val_losses
