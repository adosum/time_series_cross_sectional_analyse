import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import copy
import random
import os
from tqdm import tqdm  # <--- Imported tqdm

# ==========================================
# 0. REPRODUCIBILITY
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def augment_financial_data(X_tensor, noise_std=0.05, mask_prob=0.10):
    X_aug = X_tensor.clone()
    
    # randn_like automatically knows to use the GPU because X_aug is on the GPU
    noise = torch.randn_like(X_aug) * noise_std
    X_aug = X_aug + noise
    
    # ⭐️ Explicitly pass the device flag to torch.rand and torch.empty!
    mask = (torch.rand(X_aug.shape[0], 1, X_aug.shape[2], device=X_aug.device) > mask_prob).float()
    X_aug = X_aug * mask
    
    scale = torch.empty(X_aug.shape[0], 1, 1, device=X_aug.device).uniform_(0.95, 1.05)
    X_aug = X_aug * scale
    
    return X_aug
# ==========================================
# 1. HELPER: CREATE 3D SEQUENCES
# ==========================================
def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len - 1])
    return np.array(X_seq), np.array(y_seq)

# ==========================================
# 2. LOAD AND PREPARE DATA (3-WAY SPLIT)
# ==========================================
print("Loading MTF ML Data...")
df = pd.read_csv("USDCNH_MTF_ML_ready.csv", index_col="Datetime", parse_dates=True)

feature_cols = [
    'Log_Returns', 'AO_15m', 'MFI_10', 'MFI_14', 'MFI_20', 
    'Vol_Ratio', 'BB_PctB', 'Donchian_Pos_20', 
    'ADX_4H', 'AO_4H', 'ADX_1D', 'AO_1D'
]

XX = df[feature_cols].values
y = df['Target'].values.reshape(-1, 1)

train_idx = int(len(XX) * 0.70)
val_idx = int(len(XX) * 0.85)

XX_train = XX[:train_idx]
XX_val   = XX[train_idx:val_idx]
XX_test  = XX[val_idx:]

y_train_raw = y[:train_idx]
y_val_raw   = y[train_idx:val_idx]
y_test_raw  = y[val_idx:]

scaler = StandardScaler()
XX_train_scaled = scaler.fit_transform(XX_train)
XX_val_scaled   = scaler.transform(XX_val)
XX_test_scaled  = scaler.transform(XX_test)

SEQUENCE_LENGTH = 10
X_train_3d, y_train_aligned = create_sequences(XX_train_scaled, y_train_raw, seq_len=SEQUENCE_LENGTH)
X_val_3d,   y_val_aligned   = create_sequences(XX_val_scaled, y_val_raw, seq_len=SEQUENCE_LENGTH)
X_test_3d,  y_test_aligned  = create_sequences(XX_test_scaled, y_test_raw, seq_len=SEQUENCE_LENGTH)

device = torch.device("cuda")

X_train_tensor = torch.tensor(X_train_3d, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_aligned, dtype=torch.float32).to(device)
X_val_tensor   = torch.tensor(X_val_3d, dtype=torch.float32).to(device)
y_val_tensor   = torch.tensor(y_val_aligned, dtype=torch.float32).to(device)
X_test_tensor  = torch.tensor(X_test_3d, dtype=torch.float32).to(device)
y_test_tensor  = torch.tensor(y_test_aligned, dtype=torch.float32).to(device)
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 2.5 CREATE DATALOADERS (MINI-BATCHING)
# ==========================================
BATCH_SIZE = 64  # 64, 128, or 256 are standard for Transformers

# Wrap the tensors in a PyTorch Dataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create the DataLoaders
# We shuffle the training data so the model doesn't memorize the chronological order!
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Validation and Test sets do NOT need shuffling
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Batches per Epoch: {len(train_loader)}")

# ==========================================
# 3. BUILD THE TRANSFORMER ARCHITECTURE
# ==========================================
class USDCNH_Transformer(nn.Module):
    def __init__(self, input_size, seq_len, d_model=16, nhead=2, num_layers=2, dropout=0.2):
        super(USDCNH_Transformer, self).__init__()
        self.feature_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(d_model, 16)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = self.feature_projection(x)
        x = x + self.pos_encoder
        x = self.transformer(x)
        last_step_output = x[:, -1, :] 
        
        out = self.relu(self.fc1(last_step_output))
        out = self.dropout_layer(out)
        out = self.output(out)
        return out


# ==========================================
# 4. DEFINE THE PBT AGENT
# ==========================================
class PBTAgent:
    def __init__(self, input_size, seq_len, lr, dropout, weight_decay):
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        
        # ⭐️ Build the models and immediately send them to the GPU
        self.active_model = USDCNH_Transformer(input_size, seq_len, dropout=self.dropout).to(device)
        self.ema_model = copy.deepcopy(self.active_model).to(device)
        
        self.optimizer = optim.AdamW(
            self.active_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.fitness = 0.0

    def update_ema(self, decay=0.95):
        with torch.no_grad():
            for ema_param, active_param in zip(self.ema_model.parameters(), self.active_model.parameters()):
                ema_param.data.mul_(decay).add_(active_param.data, alpha=1.0 - decay)

# ==========================================
# 5. INITIALIZE PBT POPULATION
# ==========================================
POPULATION_SIZE = 15
GENERATIONS = 50
EPOCHS_PER_GEN = 50

print("\nInitializing PBT Population...")
population = []

for i in range(POPULATION_SIZE):
    # 1. Generate the safe, lower-bounded hyperparameters
    lr = random.uniform(0.00001, 0.0003) 
    dropout = random.uniform(0.05, 0.20) 
    wd = random.uniform(1e-5, 1e-4)
    
    # 2. CREATE the Agent object using those parameters
    agent = PBTAgent(
        input_size=len(feature_cols), 
        seq_len=SEQUENCE_LENGTH, 
        lr=lr, 
        dropout=dropout, 
        weight_decay=wd
    )
    
    # 3. APPEND the Agent to the population array! (This is what was missing)
    population.append(agent)

loss_function = nn.BCEWithLogitsLoss()

# ==========================================
# 6. CONTINUOUS PBT EVOLUTION LOOP
# ==========================================
print("\nStarting Continuous PBT Evolution...")

# 1. This creates the main progress bar AND defines 'gen'
pbar = tqdm(range(1, GENERATIONS + 1), desc="PBT Evolution", unit="gen")

for gen in pbar:
    
    # ------------------------------------------
    # STEP A: CONTINUOUS TRAINING (Mini-Batched)
    # ------------------------------------------
    agent_pbar = tqdm(population, desc=f"Gen {gen} Training", leave=False, unit="agent")
    
    for agent in agent_pbar:
        agent.active_model.train()
        
        for epoch in range(EPOCHS_PER_GEN):
            
            # ⭐️ Iterate through the mini-batches!
            for batch_X, batch_y in train_loader:
                agent.optimizer.zero_grad()
                
                # Apply Data Augmentation strictly to this tiny batch
                batch_X_aug = augment_financial_data(batch_X, noise_std=0.05, mask_prob=0.10)
                
                # Forward pass on the batch
                predictions = agent.active_model(batch_X_aug)
                loss = loss_function(predictions, batch_y)
                
                # Backprop
                loss.backward()
                
                # Gradient Clipping (Crucial for Transformers)
                torch.nn.utils.clip_grad_norm_(agent.active_model.parameters(), max_norm=1.0)
                
                # Update weights
                agent.optimizer.step()
            
            # ⭐️ Update EMA Shadow Model at the end of every epoch
            agent.update_ema(decay=0.80)
            
            # Update the inner bar with the final batch loss of the epoch
            agent_pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

    # ------------------------------------------
    # STEP B: EVALUATION (Mini-Batched)
    # ------------------------------------------
    for agent in population:
        agent.ema_model.eval()
        
        total_val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            # ⭐️ Iterate through the Validation DataLoader safely!
            for batch_X_val, batch_y_val in val_loader:
                
                # Predict on just this tiny batch
                val_predictions = agent.ema_model(batch_X_val)
                
                # PyTorch calculates the mean loss for THIS batch automatically
                batch_loss = loss_function(val_predictions, batch_y_val)
                
                # Accumulate the loss and count the batches
                total_val_loss += batch_loss.item()
                num_batches += 1
        
        # ⭐️ Calculate the true average loss across the entire validation set
        avg_val_loss = total_val_loss / num_batches
        
        # Fitness is based on the average VALIDATION loss
        agent.fitness = 1.0 / (avg_val_loss + 1e-8)

    # Sort population from best to worst
    population.sort(key=lambda x: x.fitness, reverse=True)
    best_agent = population[0]
    
    # Update the MAIN progress bar with the best stats
    pbar.set_postfix({
        'Val_Fit': f"{best_agent.fitness:.3f}",
        'LR': f"{best_agent.lr:.5f}",
        'Drop': f"{best_agent.dropout:.2f}"
    })

    # ------------------------------------------
    # STEP C: EXPLOIT AND EXPLORE
    # ------------------------------------------
    # ⭐️ FIX 3: Give them a 4-generation (200 epoch) grace period to warm up!
    GRACE_PERIOD_GENS = 4 
    
    if gen > GRACE_PERIOD_GENS and gen < GENERATIONS:
        half_pop = POPULATION_SIZE // 2
        
        for i in range(half_pop, POPULATION_SIZE):
            winner = population[i - half_pop] 
            loser = population[i]
            
            # Exploit
            loser.active_model.load_state_dict(winner.ema_model.state_dict())
            loser.ema_model.load_state_dict(winner.ema_model.state_dict())
            
            # Explore (Mutate)
            loser.lr = winner.lr * random.choice([0.8, 1.2])
            loser.dropout = max(0.1, min(0.5, winner.dropout * random.choice([0.8, 1.2])))
            loser.weight_decay = winner.weight_decay * random.choice([0.8, 1.2])
            
            # Rebuild Optimizer
            loser.optimizer = optim.AdamW(
                loser.active_model.parameters(), lr=loser.lr, weight_decay=loser.weight_decay
            )

# ==========================================
# 7. FINAL TEST ON BLIND OUT-OF-SAMPLE DATA
# ==========================================
print("\n========================================")
print("PBT Training Complete. Evaluating Ultimate Master Model...")
print("========================================")

ultimate_model = population[0].ema_model

ultimate_model.eval()
with torch.no_grad():
    test_predictions = ultimate_model(X_test_tensor) 
    test_loss = loss_function(test_predictions, y_test_tensor)
    
    binary_predictions = (test_predictions >= 0.0).float()
    correct = (binary_predictions == y_test_tensor).sum().item()
    accuracy = correct / len(y_test_tensor)

print(f"Ultimate Test Loss (BCE): {test_loss.item():.4f}")
print(f"Ultimate Test Accuracy:   {accuracy * 100:.2f}%")

torch.save(ultimate_model.state_dict(), "USDCNH_PBT_EMA_Production.pth")
print("\nMaster Model saved to 'USDCNH_PBT_EMA_Production.pth'.")