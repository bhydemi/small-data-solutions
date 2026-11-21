from TestModel import test_model
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_and_standardize_data(path):
    df = pd.read_csv(path, sep=',')
    df = df.fillna(-99)
    df = df.values.reshape(-1, df.shape[1]).astype('float32')
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)   
    return X_train, X_test, scaler

class DataBuilder(Dataset):
    def __init__(self, path, train=True):
        self.X_train, self.X_test, self.standardizer = load_and_standardize_data(path)
        if train:
            self.x = torch.from_numpy(self.X_train)
            self.len=self.x.shape[0]
        else:
            self.x = torch.from_numpy(self.X_test)
            self.len=self.x.shape[0]
        del self.X_train
        del self.X_test 
    def __getitem__(self,index):      
        return self.x[index]
    def __len__(self):
        return self.len

class CustomLoss(nn.Module):
    def __init__(self, beta=0.02):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.beta = beta
    
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + self.beta * loss_KLD

class Autoencoder(nn.Module):
    def __init__(self,D_in,H=96,H2=32,latent_dim=10):
        #Encoder
        super(Autoencoder,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2=nn.Linear(H,H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3=nn.Linear(H2,H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)
        
        # Latent vectors
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)
        
        # Decoder
        self.linear4=nn.Linear(H2,H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5=nn.Linear(H2,H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6=nn.Linear(H,D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        
        return r1, r2
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def generate_fake(mu, logvar, no_samples, scaler, model):
    # Sample from the empirical mixture of encoded posteriors. Averaging all
    # means and variances first collapses the latent distribution and produces
    # synthetic rows with unrealistically little variation.
    device = next(model.parameters()).device
    mu, logvar = mu.to(device), logvar.to(device)
    indices = torch.randint(mu.shape[0], (no_samples,), device=device)
    selected_mu = mu[indices]
    selected_std = torch.exp(0.5 * logvar[indices])
    z = selected_mu + torch.randn_like(selected_std) * selected_std
    with torch.no_grad():
        pred = model.decode(z).cpu().numpy()
    fake_data = scaler.inverse_transform(pred)
    return fake_data


def train_vae(model, trainloader, valloader, criterion, optimizer, device, num_epochs=750):
    """Train and validate the VAE every epoch; restore the best validation state."""
    best_val = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in trainloader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(batch)
            loss = criterion(reconstruction, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valloader:
                batch = batch.to(device)
                reconstruction, mu, logvar = model(batch)
                val_loss += criterion(reconstruction, batch, mu, logvar).item()

        mean_train = train_loss / max(1, len(trainloader))
        mean_val = val_loss / max(1, len(valloader))
        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)
        if mean_val < best_val:
            best_val = mean_val
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}: train loss {mean_train:.2f}; validation loss {mean_val:.2f}")

    if best_state is None:
        raise RuntimeError("VAE did not complete a validation epoch; check dataset size and batch size")
    model.load_state_dict(best_state)
    model.to(device).eval()
    # Draw synthetic records from training-set posteriors. Validation rows are
    # reserved for model selection and must not seed generated observations.
    mu_batches, logvar_batches = [], []
    with torch.no_grad():
        for batch in trainloader:
            batch = batch.to(device)
            mu, logvar = model.encode(batch)
            mu_batches.append(mu.cpu())
            logvar_batches.append(logvar.cpu())
    best_mu, best_logvar = torch.cat(mu_batches), torch.cat(logvar_batches)
    return model, best_mu, best_logvar, history

# Run the VAE training and evaluation workflow with this module.
def main():
    data_dir = Path(__file__).resolve().parent / "data"
    original_path = data_dir / "loan_continuous.csv"
    status1_path = data_dir / "loan_status_1.csv"
    expanded_path = data_dir / "loan_continuous_expanded.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    original = pd.read_csv(original_path).fillna(-99)
    status1 = original.loc[original["Loan Status"] == 1].copy()
    if status1.empty:
        raise ValueError("No Loan Status = 1 examples found")
    status1.to_csv(status1_path, index=False)
    print("Baseline on original data:")
    test_model(str(original_path))

    train_data = DataBuilder(str(status1_path), train=True)
    val_data = DataBuilder(str(status1_path), train=False)
    trainloader = DataLoader(train_data, batch_size=512, shuffle=True, drop_last=True)
    valloader = DataLoader(val_data, batch_size=512, shuffle=False, drop_last=True)
    model = Autoencoder(D_in=train_data.x.shape[1], H=96, H2=32, latent_dim=10).to(device)
    criterion = CustomLoss(beta=0.02)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model, best_mu, best_logvar, history = train_vae(
        model, trainloader, valloader, criterion, optimizer, device, num_epochs=750
    )
    pd.DataFrame(history).to_csv(data_dir / "vae_training_history.csv", index_label="epoch")
    fake_data = generate_fake(best_mu, best_logvar, 50000, train_data.standardizer, model)
    synthetic = pd.DataFrame(fake_data, columns=status1.columns)
    # Restore the known class label because the decoder also approximates that column.
    synthetic["Loan Status"] = 1
    real_std = status1.std(numeric_only=True)
    fake_std = synthetic[status1.columns].std(numeric_only=True)
    std_ratio = (fake_std / real_std.replace(0, np.nan)).dropna()
    print(f"Synthetic/real feature standard-deviation ratio: median={std_ratio.median():.2f}, "
          f"10th-90th percentile={std_ratio.quantile(0.10):.2f}-{std_ratio.quantile(0.90):.2f}")
    original["Synthetic"] = False
    synthetic["Synthetic"] = True
    expanded = pd.concat([original, synthetic], ignore_index=True)
    expanded.to_csv(expanded_path, index=False)
    print(f"Saved {len(synthetic):,} synthetic Loan Status = 1 rows to {expanded_path}")
    print("Evaluation using 25% of synthetic rows (12,500):")
    test_model(str(expanded_path), synthetic_fraction=0.25)
    print("Evaluation using all 50,000 synthetic rows:")
    test_model(str(expanded_path), synthetic_fraction=1.0)

if __name__ == '__main__':
    main()
    print("done")
