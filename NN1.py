import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Define the Neural Network Model with Dropout to avoid overfitting
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        
        # Define layers: 7 input features, 3 hidden layers, 1 output
        self.fc1 = nn.Linear(7, 64)  # First hidden layer: 7 inputs, 64 hidden units
        self.fc2 = nn.Linear(64, 128)  # Second hidden layer: 64 inputs, 128 hidden units
        self.fc3 = nn.Linear(128, 64)  # Third hidden layer: 128 inputs, 64 hidden units
        self.output = nn.Linear(64, 1)  # Output layer: 64 inputs, 1 output
        
        # Activation function
        self.relu = nn.ReLU()

        # Dropout layers to avoid overfitting
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # Define forward pass with dropout
        x = self.relu(self.fc1(x))  # First hidden layer + activation
        x = self.dropout(x)  # Dropout after first hidden layer
        x = self.relu(self.fc2(x))  # Second hidden layer + activation
        x = self.dropout(x)  # Dropout after second hidden layer
        x = self.relu(self.fc3(x))  # Third hidden layer + activation
        x = self.output(x)  # Output layer
        return x

# Function to process each row (train or test a portion of the dataset)
def train(rank, model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float(), target.float()

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)

        # Backward pass and optimization
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        running_loss += loss.item()
    print(f"Processor {rank}, Epoch {epoch}, Loss: {running_loss/len(train_loader)}")

def main_process(x):
    # Read CSV data (assuming it's stored in 'data.csv')
    data = pd.read_csv('data.csv')

    # Extracting labels and features (Columns 4 to 10 are the features, 1st column is the label)
    labels = data.iloc[:, 0].values  # Labels (first column)
    features = data.iloc[:, 3:10].values  # Features (columns 4 to 10)

    # Convert to torch tensors
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)  # Reshape labels to match output

    # Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Create DataLoader for batches
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create a model instance
    model = SimpleNN()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Mean Squared Error loss
    criterion = nn.MSELoss()

    # Sharing the model across processes
    model.share_memory()

    # Multiprocessing setup
    processes = []
    epochs = 10  # Number of training epochs

    # Start x processes
    for rank in range(x):
        for epoch in range(epochs):
            p = mp.Process(target=train, args=(rank, model, train_loader, criterion, optimizer, epoch))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    x = 4  # Number of processors you want to use
    mp.set_start_method('spawn')  # For safe multiprocessing
    main_process(x)
