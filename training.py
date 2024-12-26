import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from time import time
from classifier import CNNClassifier

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

transform_with_augmentation = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.EMNIST(split="balanced", root="./data", train = True, transform=transform, download=True)
test_dataset = datasets.EMNIST(split="balanced", root="./data", train=False, transform=transform, download=True)

model = CNNClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training

batch_size = 64

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    correct_train, total_train = 0, 0
    start = time()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        #Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy_train = correct_train / total_train

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy_train * 100:.2f}%, Time: {time() - start}s')

torch.save(model.state_dict(), 'model_state_dict.pth')

model.eval()  # Set the model to evaluation mode
correct_test, total_test = 0, 0
all_predicted, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) #we look for the label with the highest assigned probability for each image
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

        # Collect predictions and true labels for later analysis
        all_predicted.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
accuracy_test = 0
for i in range(len(all_labels)):
    if (all_labels[i] == all_predicted[i]):
        accuracy_test += 1
accuracy_test /= len(all_labels)

# Print test accuracy
print(f'Test Accuracy: {accuracy_test * 100:.2f}%')

# Calculate and print confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predicted)
print('\nConfusion Matrix:')
print(conf_matrix)

# Calculate and print classification report
class_report = classification_report(all_labels, all_predicted)
print('\nClassification Report:')
print(class_report)

def visualize_results_adv(model, test_loader, num_images=5):
    model.eval()
    images_shown = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            if images_shown >= num_images:
                break

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            # if (predicted[0] == labels[0]):
            #     continue
            # Convert the image tensor to a NumPy array
            image = images[0][0].squeeze().cpu().numpy()

            # Visualize the test images and their predictions
            plt.subplot(1, num_images, images_shown + 1)
            plt.imshow(image, cmap='gray')  # Use the NumPy array for imshow
            plt.title(f'Predicted: {predicted[0]}\nActual: {labels[0]}')
            plt.axis('off')

            images_shown += 1

    plt.show()
visualize_results_adv(model, test_loader)