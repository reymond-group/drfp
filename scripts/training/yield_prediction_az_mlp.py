import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def save_results(
    set_name: str,
    split_id: str,
    file_name: str,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> None:
    with open(f"{set_name}_{split_id}_{file_name}.csv", "w+") as f:
        for gt, pred in zip(ground_truth, prediction):
            f.write(f"{set_name},{split_id},{file_name},{gt},{pred}\n")


def predict_az():
    root_path = Path(__file__).resolve().parent
    az_file_path = Path(root_path, "../../data/az/az-10240-3-true.pkl")

    data = pickle.load(open(az_file_path, "rb"))

    for i, split in enumerate(data):
        X_train, y_train, X_val, y_val, X_test, y_test = (
            torch.FloatTensor(split["train"]["X"]),
            torch.FloatTensor(split["train"]["y"]),
            torch.FloatTensor(split["valid"]["X"]),
            torch.FloatTensor(split["valid"]["y"]),
            torch.FloatTensor(split["test"]["X"]),
            torch.FloatTensor(split["test"]["y"]),
        )

        batch_size = 32
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        num_epochs = 1000
        best_val_loss = float("inf")

        model = MLPRegressor(input_dim=X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).flatten()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    val_outputs = model(batch_X).flatten()
                    val_loss += criterion(val_outputs, batch_y).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_az_mlp_tmp.pt")

            if (epoch + 1) % 10 == 0:
                print(
                    f"Split {i+1}/{len(data)} Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}"
                )

        model.load_state_dict(torch.load("best_az_mlp_tmp.pt"))

        model.eval()
        test_outputs = []

        with torch.no_grad():
            for batch_X, _ in test_loader:
                test_outputs.extend(model(batch_X).flatten().numpy())

        test_outputs = torch.tensor(test_outputs)
        test_outputs[test_outputs < 0.0] = 0.0

        r2 = r2_score(y_test, test_outputs)
        mae = mean_absolute_error(y_test, test_outputs)
        print(f"Test R2 Score: {r2:.4f}, Test MAE: {mae:.4f}")


def main():
    predict_az()


if __name__ == "__main__":
    main()
