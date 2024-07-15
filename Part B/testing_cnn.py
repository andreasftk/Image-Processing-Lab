import requests
from PIL import Image, ImageOps
from io import BytesIO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(6, 32, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 4 * 4)
        features = self.relu(self.fc1(x))
        logits = self.fc2(features)
        return logits, features

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Load the image from the URL
# url = 'http://calstormbasketball.com/wp-content/uploads/2018/08/5020657994731_01c.jpeg'
url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSWqArjoWMekSCJ3RuuA2oE1oPwU-D1EKIZoQ&s'
# url = 'https://raw.githubusercontent.com/tphinkle/tphinkle.github.io/master/images/2017-8-15/mnist_0.png'
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale

# Invert the image to make background black and text white
img = ImageOps.invert(img)

# Process the image to match MNIST format (28x28)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

img = transform(img).unsqueeze(0)  # Add batch dimension

# Predict the digit
with torch.no_grad():
    logits, features = model(img)
    prediction = torch.argmax(logits, dim=1).item()

print(f'Predicted digit: {prediction}')

# Display the image
plt.imshow(img.squeeze().numpy(), cmap='gray')
plt.title(f'Predicted: {prediction}')
plt.axis('off')
plt.show()
