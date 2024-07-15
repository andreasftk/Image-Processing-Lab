import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Step 1: Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Step 2: Display a sample image from each class
fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i in range(10):
    img, label = mnist_trainset.data[mnist_trainset.targets == i][0], i
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(label)
    axes[i].axis('off')
plt.show()

# Step 3: Split the dataset into training and validation sets (80%-20%)
train_size = int(0.8 * len(mnist_trainset))
val_size = len(mnist_trainset) - train_size
mnist_trainset, mnist_valset = torch.utils.data.random_split(mnist_trainset, [train_size, val_size])

train_dataloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(mnist_valset, batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(mnist_testset, batch_size=32, shuffle=False)

# Step 4: Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5, padding=0)  # Changed input channels to 6
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(12 * 4 * 4, 128)  # Adjusted to match the reduced spatial dimensions after pooling
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 12 * 4 * 4)  # Adjusted to match the reduced spatial dimensions after pooling
        features = self.relu(self.fc1(x))
        logits = self.fc2(features)
        return logits, features


model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 5: Train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    train_loss, val_loss = [], []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()           # Clear previous gradients
            outputs, _ = model(images)      # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()                 # Backward pass to compute gradients
            optimizer.step()                # Update parameters using gradients
            running_train_loss += loss.item()

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_val_loss = running_val_loss / len(val_loader)
        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

    return train_loss, val_loss

train_loss, val_loss = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=10)

# Step 6: Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), train_loss, label='Train Loss')
plt.plot(range(1, 11), val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Step 7: Evaluate the model on the test set
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

results = list()
total = 0

for itr, (image, label) in enumerate(test_dataloader):
    if torch.cuda.is_available():
        image = image.cuda()
        label = label.cuda()

    pred, _ = model(image)
    pred = torch.nn.functional.softmax(pred, dim=1)

    for i, p in enumerate(pred):
        if label[i] == torch.max(p.data, 0)[1]:
            total += 1
            results.append((image[i], torch.max(p.data, 0)[1]))

test_accuracy = total / len(mnist_testset)
print('Test accuracy {:.8f}'.format(test_accuracy))

# Visualize results
fig = plt.figure(figsize=(20, 10))

for i in range(1, 21):
    if i-1 < len(results):
        
        img = results[i-1][0].squeeze(0).detach().cpu()
  
        img = transforms.ToPILImage(mode='L')(img)
        fig.add_subplot(4, 5, i)  
        plt.title(results[i-1][1].item())
        plt.imshow(img, cmap='gray')
        plt.axis('off')  
    else:
        break

plt.show()

# Initialize lists to collect true and predicted labels
true_labels = []
pred_labels = []

# Set the model to evaluation mode
model.eval()

# Evaluate the model on the test dataset
with torch.no_grad():
    for images, labels in test_dataloader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        # Get model predictions
        outputs, _ = model(images)
        _, preds = torch.max(outputs, 1)

        # Extend the lists with true and predicted labels
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, pred_labels)

# Function to plot confusion matrix with all values
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(ticks=[0.5 + i for i in range(10)], labels=[str(i) for i in range(10)], rotation=0)
    plt.yticks(ticks=[0.5 + i for i in range(10)], labels=[str(i) for i in range(10)], rotation=0)

    # Add all values of confusion matrix
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j + 0.5, i + 0.5, conf_matrix[i, j], ha='center', va='center', color='orange', fontsize=10)

    plt.show()

# Plot the confusion matrix with all values
plot_confusion_matrix(conf_matrix)

# Print the confusion matrix for debugging
print('Confusion Matrix:')
print(conf_matrix)



# Initialize lists to collect true and predicted labels
true_labels = []
pred_labels = []

# Step 8: Repeat training with different proportions of training data and analyze performance
total_train_size = len(mnist_trainset)
proportions = [0.05, 0.1, 0.5, 1.0]
results = {}
all_train_losses = []
all_val_losses = []

for prop in proportions:
    print(f'\nTraining with {prop * 100}% of training data...')
    subset_size = int(prop * total_train_size)
    # Randomly select subset from mnist_trainset
    subset_indices = np.random.choice(total_train_size, subset_size, replace=False)
    subset_trainset = torch.utils.data.Subset(mnist_trainset, subset_indices)
    print(len(subset_trainset))
    
    train_size = int(0.8 * subset_size)  # 80% for training
    val_size = subset_size - train_size  # 20% for validation

    subset_trainset, subset_valset = torch.utils.data.random_split(subset_trainset, [train_size, val_size])
    print(len(subset_trainset))
    print(len(subset_valset))
    subset_train_loader = torch.utils.data.DataLoader(subset_trainset, batch_size=32, shuffle=True)
    subset_val_loader = torch.utils.data.DataLoader(subset_valset, batch_size=32, shuffle=False)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_loss, val_loss = train_model(model, subset_train_loader, subset_val_loader, criterion, optimizer, epochs=10)
    all_train_losses.append(train_loss)
    all_val_losses.append(val_loss)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    test_correct = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs, _ = model(images)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    test_accuracy = test_correct / len(mnist_testset)
    results[prop] = test_accuracy
    print(f'Test Accuracy with {prop * 100}% of training data: {test_accuracy:.4f}')

    conf_matrix = confusion_matrix(true_labels, pred_labels)
    plot_confusion_matrix(conf_matrix)
    print(conf_matrix)

    true_labels.clear()
    pred_labels.clear()

# Plot training and validation losses for different proportions
plt.figure(figsize=(15, 10))
epochs = range(1, 11)
for i, prop in enumerate(proportions):
    plt.plot(epochs, all_train_losses[i], label=f'Train Loss ({prop * 100}%)', linestyle='-')
    plt.plot(epochs, all_val_losses[i], label=f'Validation Loss ({prop * 100}%)', linestyle='-.')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for Different Proportions of Training Data')
plt.legend()
plt.show()


# Plot performance vs. training data size
plt.figure(figsize=(10, 5))
plt.plot([prop * 100 for prop in proportions], [results[prop] for prop in proportions], marker='o')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs. Training Data Size')
plt.show()



