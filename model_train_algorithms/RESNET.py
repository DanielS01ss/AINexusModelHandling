import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import mlflow
import mlflow.pytorch
import pickle
import uuid

# Define ResNet architecture
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Load your dataset and preprocess it
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize ResNet model
resnet_model = SimpleResNet()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)

# Train the model
with mlflow.start_run():
    for epoch in range(5):  # Adjust the number of epochs as needed
        for inputs, labels in train_loader:
            # Forward pass
            outputs = resnet_model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Log the model and metrics with MLflow
    mlflow.pytorch.log_model(resnet_model, "resnet_model")
    mlflow.log_param("num_epochs", 5)  # Log other parameters as needed
    mlflow.log_metric("final_loss", loss.item())

    # Save the model as a pickle file
    model_pkl_path ="{}_random_forest_model.pkl".format(uuid.uuid4())
    with open(model_pkl_path, "wb") as model_file:
        pickle.dump(resnet_model, model_file)

