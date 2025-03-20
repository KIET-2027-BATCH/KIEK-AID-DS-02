import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"C:\Users\kurra\OneDrive\Attachments\csv\backend\hello.py\electricity_data_cleaned.csv")

# Feature scaling
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# Prepare data for LSTM
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

seq_length = 30  # Use past 30 days for prediction
X, y = create_sequences(data_scaled, seq_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Taking the last output
        return out

# Initialize model
input_size = X.shape[2]  # Number of features
model = LSTMModel(input_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
batch_size = 16

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred_train = model(X_train_torch)
    loss = criterion(y_pred_train, y_train_torch)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# Evaluate model
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_torch)

# Convert predictions back to original scale
y_pred_rescaled = scaler.inverse_transform(y_pred_test.numpy())
y_test_rescaled = scaler.inverse_transform(y_test)

# Plot results
plt.figure(figsize=(12,6))
plt.plot(y_test_rescaled, label='Actual Consumption')
plt.plot(y_pred_rescaled, label='Predicted Consumption')
plt.legend()
plt.title('Electricity Consumption Forecasting')
plt.show()

# Save model
torch.save(model.state_dict(), 'electricity_forecast_model.pth')
