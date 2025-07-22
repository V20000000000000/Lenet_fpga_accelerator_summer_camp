import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


##########################################################################
############################## LeNet5 model ##############################
##########################################################################
class LeNet5(nn.Module): 
    def __init__(self):
        super().__init__()
        # Feature extractor: Convolutional and pooling layers
        self.features = nn.Sequential(
            # Input: 1x28x28 (MNIST image size)
            # The original LeNet-5 used 32x32. Padding=2 adapts it for 28x28.
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),   
            nn.ReLU(),
            # Output: 6x28x28
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Output: 6x14x14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            # Output: 16x10x10
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Output: 16x5x5
        )
        # Classifier: Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),

            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        x = self.features(x) # Pass through conv layers
        x = self.classifier(x) # Pass through fully-connected layers
        return x


##########################################################################
############################ data preprocessing ##########################
##########################################################################
# DataSet (MNIST)
# Define the transformations for the dataset(img to tensor, normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

# download and load the MNIST train dataset
train_dataset = torchvision.datasets.MNIST(
    root = './data',
    train = True,
    download = True,
    transform = transform
)

# build the train DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# download and load the MNIST test dataset
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# build the test data loader
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)


###########################################################################
############################ training the model ###########################
###########################################################################
# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# implement the model to the device
model = LeNet5().to(device)

#define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) # move data to the device
        optimizer.zero_grad() # Zero the gradients
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, labels)
        loss.backward() # Backward pass
        optimizer.step() # Update weights
        running_loss += loss.item()
        if (i + 1) % 200 == 0: # 每 200 個 batch 印一次
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 200:.4f}')
            running_loss = 0.0

print('training finished!')

# Save the trained model's state dictionary
torch.save(model.state_dict(), 'lenet.pt')
print("Model state_dict saved to lenet.pt")


###########################################################################
########################## evaluating the model ###########################
###########################################################################
model.eval() # set the model to evaluation mode
correct = 0
total = 0

# evaluat the model does not require gradient calculation
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on the test set: {accuracy:.2f}%')