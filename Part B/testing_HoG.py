import requests
from PIL import Image, ImageOps
from io import BytesIO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
import matplotlib.pyplot as plt
from torchvision import datasets

# Define the FCN model
class FCN(nn.Module):
    def __init__(self, input_size):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the MNIST dataset and extract HoG features to determine the input size
transform = transforms.Compose([transforms.ToTensor()])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

def dataset_to_numpy(dataset):
    data = dataset.data.numpy()
    labels = dataset.targets.numpy()
    return data, labels

train_images, _ = dataset_to_numpy(mnist_trainset)

def extract_hog_features(images):
    hog_features = []
    for image in images:
        hog_feature = hog(image, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualize=False)
        hog_features.append(hog_feature)
    return np.array(hog_features)

train_hog_features = extract_hog_features(train_images[:10])  # Extract features from a small subset for dimension check
input_size = train_hog_features.shape[1]
print(f"Detected HoG feature size: {input_size}")

# Initialize the model with the correct input size
model = FCN(input_size)
model.load_state_dict(torch.load('best_fcn_model.pth'))
model.eval()

# Load the image from the URL
# url = 'http://calstormbasketball.com/wp-content/uploads/2018/08/5020657994731_01c.jpeg'
url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSWqArjoWMekSCJ3RuuA2oE1oPwU-D1EKIZoQ&s'
# url = 'https://raw.githubusercontent.com/tphinkle/tphinkle.github.io/master/images/2017-8-15/mnist_0.png'

response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale

# Invert the image to make background black and text white
img = ImageOps.invert(img)

# Resize the image to 28x28
img = img.resize((28, 28), Image.LANCZOS)

# Convert image to numpy array
img_array = np.array(img)

# Extract HoG features from the image
hog_feature = hog(img_array, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualize=False)
hog_feature = torch.tensor(hog_feature, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Predict the digit
with torch.no_grad():
    logits = model(hog_feature)
    prediction = torch.argmax(logits, dim=1).item()

print(f'Predicted digit: {prediction}')

# Display the image
plt.imshow(img, cmap='gray')
plt.title(f'Predicted: {prediction}')
plt.axis('off')
plt.show()
