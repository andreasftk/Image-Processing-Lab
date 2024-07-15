import torch
import torch.nn as nn
import torch.optim as optim
from skimage.feature import hog
from skimage.transform import resize
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

# Step 1: Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Convert MNIST dataset to numpy arrays
def dataset_to_numpy(dataset):
    data = dataset.data.numpy()
    labels = dataset.targets.numpy()
    return data, labels

train_images, train_labels = dataset_to_numpy(mnist_trainset)
test_images, test_labels = dataset_to_numpy(mnist_testset)

# Step 2: Extract HoG features
def extract_hog_features(images):
    hog_features = []
    for image in images:
        hog_feature = hog(image, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualize=False)
        hog_features.append(hog_feature)
    return np.array(hog_features)

print("Extracting Train HoG features.")
train_hog_features = extract_hog_features(train_images)
print("Extracting Test HoG features.")
test_hog_features = extract_hog_features(test_images)

# Step 3: Define the FCN model (Fully Connected Network)
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(train_hog_features.shape[1], 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Convert features and labels to PyTorch tensors
X_train, X_val, y_train, y_val = train_test_split(train_hog_features, train_labels, test_size=0.2, random_state=42)
X_train, X_val = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32)
y_train, y_val = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_val, dtype=torch.long)
X_test, y_test = torch.tensor(test_hog_features, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

model = FCN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 4: Train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    train_loss, val_loss = [], []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_val_loss = running_val_loss / len(val_loader)
        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_fcn_model.pth')

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

    return train_loss, val_loss

train_loss, val_loss = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=10)

# Step 5: Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), train_loss, label='Train Loss')
plt.plot(range(1, 11), val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Step 6: Evaluate the model on the test set
model.load_state_dict(torch.load('best_fcn_model.pth'))
model.eval()

test_correct = 0
with torch.no_grad():
    for images, labels in test_dataloader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()

test_accuracy = test_correct / len(test_dataset)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Compute the confusion matrix
true_labels, pred_labels = [], []
with torch.no_grad():
    for images, labels in test_dataloader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

conf_matrix = confusion_matrix(true_labels, pred_labels)

# Plot the confusion matrix
# Function to plot confusion matrix with all values
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix with All Values')
    plt.xticks(ticks=[0.5 + i for i in range(10)], labels=[str(i) for i in range(10)], rotation=0)
    plt.yticks(ticks=[0.5 + i for i in range(10)], labels=[str(i) for i in range(10)], rotation=0)

    # Add all values of confusion matrix
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j + 0.5, i + 0.5, conf_matrix[i, j], ha='center', va='center', color='orange', fontsize=10)

    plt.show()

# Plot the confusion matrix with all values
plot_confusion_matrix(conf_matrix)

true_labels.clear()
pred_labels.clear()

# Step 8: Repeat training with different proportions of training data and analyze performance
total_train_size = len(X_train)
proportions = [0.05, 0.1, 0.5, 1.0]
results = {}
all_train_losses = []
all_val_losses = []

for prop in proportions:
    print(f'\nTraining with {prop * 100}% of training data...')
    subset_size = int(prop * total_train_size)
    # Randomly select subset from X_train
    subset_indices = np.random.choice(total_train_size, subset_size, replace=False)
    subset_X_train = X_train[subset_indices]
    subset_y_train = y_train[subset_indices]

    train_size = int(0.8 * subset_size)  # 80% for training
    val_size = subset_size - train_size  # 20% for validation

    subset_X_train, subset_X_val, subset_y_train, subset_y_val = train_test_split(subset_X_train, subset_y_train, test_size=0.2, random_state=42)

    subset_train_dataset = torch.utils.data.TensorDataset(subset_X_train.clone().detach(), subset_y_train.clone().detach())
    subset_val_dataset = torch.utils.data.TensorDataset(subset_X_val.clone().detach(), subset_y_val.clone().detach())
    subset_train_loader = torch.utils.data.DataLoader(subset_train_dataset, batch_size=32, shuffle=True)
    subset_val_loader = torch.utils.data.DataLoader(subset_val_dataset, batch_size=32, shuffle=False)

    model = FCN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_loss, val_loss = train_model(model, subset_train_loader, subset_val_loader, criterion, optimizer, epochs=10)
    all_train_losses.append(train_loss)
    all_val_losses.append(val_loss)
    model.load_state_dict(torch.load('best_fcn_model.pth'))
    model.eval()

    test_correct = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    test_accuracy = test_correct / len(test_dataset)
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