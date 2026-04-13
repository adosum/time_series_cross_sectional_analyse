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
import matplotlib.pyplot as plt
from datetime import datetime

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

    num_features = X_aug.shape[2]

    noise_std_tensor = torch.as_tensor(noise_std, dtype=X_aug.dtype, device=X_aug.device)
    if noise_std_tensor.ndim == 0:
        noise_std_tensor = noise_std_tensor.view(1, 1, 1)
    elif noise_std_tensor.ndim == 1 and noise_std_tensor.numel() == num_features:
        noise_std_tensor = noise_std_tensor.view(1, 1, num_features)
    else:
        raise ValueError(f"noise_std must be a scalar or length-{num_features} vector")

    mask_prob_tensor = torch.as_tensor(mask_prob, dtype=X_aug.dtype, device=X_aug.device)
    if mask_prob_tensor.ndim == 0:
        mask_prob_tensor = mask_prob_tensor.view(1, 1, 1)
    elif mask_prob_tensor.ndim == 1 and mask_prob_tensor.numel() == num_features:
        mask_prob_tensor = mask_prob_tensor.view(1, 1, num_features)
    else:
        raise ValueError(f"mask_prob must be a scalar or length-{num_features} vector")
    mask_prob_tensor = mask_prob_tensor.clamp(0.0, 1.0)
    
    # randn_like automatically knows to use the GPU because X_aug is on the GPU
    noise = torch.randn_like(X_aug) * noise_std_tensor
    X_aug = X_aug + noise
    
    # ⭐️ Explicitly pass the device flag to torch.rand and torch.empty!
    mask = (torch.rand(X_aug.shape[0], 1, num_features, device=X_aug.device) > mask_prob_tensor).float()
    X_aug = X_aug * mask
    
    scale = torch.empty(X_aug.shape[0], 1, 1, device=X_aug.device).uniform_(0.95, 1.05)
    X_aug = X_aug * scale
    
    return X_aug


def append_error_feature(X_batch, error_history):
    """
    Append autoregressive error history channels.
    error_history shape: [k], where k is configurable (e.g., 4, 8).
    The k values are broadcast over all time steps in the current sequence.
    """
    if isinstance(error_history, torch.Tensor):
        err_vec = error_history.to(device=X_batch.device, dtype=X_batch.dtype).flatten()
    else:
        err_vec = torch.tensor(error_history, dtype=X_batch.dtype, device=X_batch.device).flatten()

    k = err_vec.numel()
    err_map = err_vec.view(1, 1, k).expand(X_batch.shape[0], X_batch.shape[1], k)
    return torch.cat([X_batch, err_map], dim=2)


def update_error_history(error_history, new_error):
    """Shift left and append the newest error at the end."""
    return error_history[1:] + [float(new_error)]


def compute_combined_confidence(abs_pred, direction_prob, volatility_pred):
    """
    Combine magnitude and direction confidence with volatility uncertainty.
    abs_pred: predicted absolute value
    direction_prob: sigmoid(direction_logits) soft probability
    volatility_pred: predicted volatility (uncertainty)
    """
    # Magnitude confidence: how extreme is the prediction (closer to 0 or 1)
    abs_confidence = torch.clamp(abs_pred, 0.0, 1.0)  # normalized magnitude
    
    # Direction confidence: how close to 0 or 1 (how confident in direction)
    direction_confidence = torch.clamp(
        2.0 * torch.abs(direction_prob - 0.5), 0.0, 1.0
    )  # 0 at 0.5, 1 at 0 or 1
    
    # Volatility-based confidence: lower volatility = higher confidence
    vol_confidence = volatility_to_confidence(volatility_pred)
    
    # Combine: average of all three confidence signals
    combined = (abs_confidence + direction_confidence + vol_confidence) / 3.0
    return torch.clamp(combined, 0.0, 1.0)


def volatility_to_confidence(volatility_pred):
    """Map predicted volatility to confidence in [0, 1].
    Lower volatility = higher confidence.
    """
    return torch.clamp(1.0 - torch.abs(volatility_pred), 0.0, 1.0)


def compute_trend_strength(X_batch):
    """
    Compute trend strength from sequence: mean absolute log returns over the window.
    X_batch shape: [batch_size, seq_len, num_features]
    Feature 0 is Log_Returns.
    """
    log_returns = X_batch[:, :, 0]  # [batch_size, seq_len]
    trend = torch.mean(torch.abs(log_returns), dim=1, keepdim=True)  # [batch_size, 1]
    return trend
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

noise_std_by_tf = {
    '15m': 0.0009,
    '4H': 0.0006,
    '1D': 0.0003,
}

mask_prob_by_tf = {
    '15m': 0.10,
    '4H': 0.07,
    '1D': 0.04,
}

feature_noise_std = []
feature_mask_prob = []
for col in feature_cols:
    if col.endswith('_1D'):
        tf_key = '1D'
    elif col.endswith('_4H'):
        tf_key = '4H'
    else:
        tf_key = '15m'

    feature_noise_std.append(noise_std_by_tf[tf_key])
    feature_mask_prob.append(mask_prob_by_tf[tf_key])

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Keep chronological order because the model now consumes previous prediction error.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Validation and Test sets do NOT need shuffling
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Batches per Epoch: {len(train_loader)}")

ERROR_HISTORY_LEN = 4  # set to 4, 8, etc.
MODEL_INPUT_SIZE = len(feature_cols) + ERROR_HISTORY_LEN

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
        self.abs_output = nn.Linear(16, 1)  # magnitude/absolute value prediction
        self.direction_output = nn.Linear(16, 1)  # direction classification (binary)
        self.volatility_output = nn.Linear(16, 1)  # volatility_pred (uncertainty)
        self.trend_output = nn.Linear(16, 1)  # trend_strength_pred

    def forward(self, x):
        x = self.feature_projection(x)
        x = x + self.pos_encoder
        x = self.transformer(x)
        last_step_output = x[:, -1, :] 
        
        shared = self.relu(self.fc1(last_step_output))
        shared = self.dropout_layer(shared)
        abs_logits = self.abs_output(shared)
        direction_logits = self.direction_output(shared)
        volatility_pred = self.volatility_output(shared)
        trend_pred = self.trend_output(shared)
        return abs_logits, direction_logits, volatility_pred, trend_pred


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
        input_size=MODEL_INPUT_SIZE,
        seq_len=SEQUENCE_LENGTH, 
        lr=lr, 
        dropout=dropout, 
        weight_decay=wd
    )
    
    # 3. APPEND the Agent to the population array! (This is what was missing)
    population.append(agent)

loss_function = nn.BCEWithLogitsLoss()
abs_loss_function = nn.MSELoss()  # for magnitude prediction
volatility_loss_function = nn.MSELoss()
trend_loss_function = nn.MSELoss()

ABS_LOSS_WEIGHT = 0.20  # weight for magnitude head
VOLATILITY_LOSS_WEIGHT = 0.25
TREND_LOSS_WEIGHT = 0.10

# # # ==========================================
# # # 6.1 pure model training loop with tqdm progress bars
# # # ==========================================
# model = USDCNH_Transformer(input_size=MODEL_INPUT_SIZE, seq_len=SEQUENCE_LENGTH, nhead=2, num_layers=2, d_model=32, dropout=0.2).to(device)
# optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
# EPOCHS = 100
# best_val_loss = float('inf')

# # ==========================================
# # METRICS TRACKING
# # ==========================================
# training_losses = []
# training_abs_losses = []
# training_direction_losses = []
# training_vol_losses = []
# training_trend_losses = []
# validation_losses = []
# validation_abs_losses = []
# validation_direction_losses = []
# validation_vol_losses = []
# validation_trend_losses = []
# validation_epochs = []
# test_accuracy = None
# test_confidence_mean = None
# test_accuracy_conf_55 = None
# test_coverage_conf_55 = None
# test_accuracy_conf_60 = None
# test_coverage_conf_60 = None
# test_accuracy_conf_70 = None
# test_coverage_conf_70 = None

# print("\nStarting Pure Model Training Loop...")
# for epoch in tqdm(range(1, EPOCHS + 1), desc="Training Epochs", unit="epoch"):
#     model.train()
#     epoch_loss = 0.0
#     epoch_abs_loss = 0.0
#     epoch_direction_loss = 0.0
#     epoch_vol_loss = 0.0
#     epoch_trend_loss = 0.0
#     train_error_history = [0.0] * ERROR_HISTORY_LEN
    
#     for batch_X, batch_y in train_loader:
#         optimizer.zero_grad()
        
#         # Apply Data Augmentation strictly to this tiny batch
#         batch_X_aug = augment_financial_data(
#             batch_X,
#             noise_std=feature_noise_std,
#             mask_prob=feature_mask_prob,
#         )
#         batch_X_with_error = append_error_feature(batch_X_aug, train_error_history)
        
#         abs_logits, direction_logits, volatility_pred, trend_pred = model(batch_X_with_error)
        
#         # Magnitude loss: predict absolute value of target
#         abs_target = torch.abs(batch_y)
#         abs_loss = abs_loss_function(torch.relu(abs_logits), abs_target)
        
#         # Direction loss: binary classification
#         direction_loss = loss_function(direction_logits, (batch_y > 0.5).float())

#         # Volatility target: absolute realized error (uncertainty)
#         direction_prob = torch.sigmoid(direction_logits)
#         realized_error = batch_y - direction_prob
#         volatility_target = torch.abs(realized_error)
#         vol_loss = volatility_loss_function(torch.abs(volatility_pred), volatility_target)

#         # Trend strength target: computed from non-augmented sequence for train/val/test consistency
#         trend_target = compute_trend_strength(batch_X)
#         trend_loss = trend_loss_function(torch.relu(trend_pred), trend_target)

#         loss = direction_loss + ABS_LOSS_WEIGHT * abs_loss + VOLATILITY_LOSS_WEIGHT * vol_loss + TREND_LOSS_WEIGHT * trend_loss
        
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()

#         # Keep online error memory: e_t = binary_target - p_t, carried to the next sequence.
#         with torch.no_grad():
#             last_prob = torch.sigmoid(direction_logits[-1]).item()
#             last_binary_target = (batch_y[-1] > 0.5).float().item()  # Binarize for consistency
#             newest_error = last_binary_target - last_prob
#             train_error_history = update_error_history(train_error_history, newest_error)
        
#         epoch_loss += loss.item()
#         epoch_abs_loss += abs_loss.item()
#         epoch_direction_loss += direction_loss.item()
#         epoch_vol_loss += vol_loss.item()
#         epoch_trend_loss += trend_loss.item()
    
#     avg_epoch_loss = epoch_loss / len(train_loader)
#     avg_epoch_abs_loss = epoch_abs_loss / len(train_loader)
#     avg_epoch_direction_loss = epoch_direction_loss / len(train_loader)
#     avg_epoch_vol_loss = epoch_vol_loss / len(train_loader)
#     avg_epoch_trend_loss = epoch_trend_loss / len(train_loader)
#     training_losses.append(avg_epoch_loss)
#     training_abs_losses.append(avg_epoch_abs_loss)
#     training_direction_losses.append(avg_epoch_direction_loss)
#     training_vol_losses.append(avg_epoch_vol_loss)
#     training_trend_losses.append(avg_epoch_trend_loss)
#     tqdm.write(f"Epoch {epoch}/{EPOCHS} - Total: {avg_epoch_loss:.4f} (abs={avg_epoch_abs_loss:.4f}, dir={avg_epoch_direction_loss:.4f}, vol={avg_epoch_vol_loss:.4f}, trend={avg_epoch_trend_loss:.4f})")
    
#     # evalute for every 10 epochs
#     if epoch % 10 == 0:
#         model.eval()
#         with torch.no_grad():
#             val_error_history = [0.0] * ERROR_HISTORY_LEN
#             total_val_abs = 0.0
#             total_val_direction = 0.0
#             total_val_vol = 0.0
#             total_val_trend = 0.0
#             total_val_count = 0

#             for batch_X_val, batch_y_val in val_loader:
#                 batch_X_val_with_error = append_error_feature(batch_X_val, val_error_history)
#                 val_abs_batch, val_direction_batch, val_vol_pred_batch, val_trend_pred_batch = model(batch_X_val_with_error)

#                 # Magnitude loss
#                 val_abs_target = torch.abs(batch_y_val)
#                 batch_abs = abs_loss_function(torch.relu(val_abs_batch), val_abs_target)
                
#                 # Direction loss
#                 batch_direction = loss_function(val_direction_batch, (batch_y_val > 0.5).float())
                
#                 # Volatility loss
#                 val_direction_prob = torch.sigmoid(val_direction_batch)
#                 batch_realized_err = batch_y_val - val_direction_prob
#                 batch_vol_target = torch.abs(batch_realized_err)
#                 batch_vol = volatility_loss_function(torch.abs(val_vol_pred_batch), batch_vol_target)
                
#                 # Trend loss
#                 batch_trend_target = compute_trend_strength(batch_X_val)
#                 batch_trend = trend_loss_function(torch.relu(val_trend_pred_batch), batch_trend_target)
                
#                 bs = batch_X_val.shape[0]
#                 total_val_abs += batch_abs.item() * bs
#                 total_val_direction += batch_direction.item() * bs
#                 total_val_vol += batch_vol.item() * bs
#                 total_val_trend += batch_trend.item() * bs
#                 total_val_count += bs

#                 last_prob = torch.sigmoid(val_direction_batch[-1]).item()
#                 last_binary_target = (batch_y_val[-1] > 0.5).float().item()  # Binarize for consistency
#                 newest_error = last_binary_target - last_prob
#                 val_error_history = update_error_history(val_error_history, newest_error)

#             val_abs_loss = total_val_abs / total_val_count
#             val_direction_loss = total_val_direction / total_val_count
#             val_vol_loss = total_val_vol / total_val_count
#             val_trend_loss = total_val_trend / total_val_count
#             val_loss = val_direction_loss + ABS_LOSS_WEIGHT * val_abs_loss + VOLATILITY_LOSS_WEIGHT * val_vol_loss + TREND_LOSS_WEIGHT * val_trend_loss
#             validation_losses.append(val_loss)
#             validation_abs_losses.append(val_abs_loss)
#             validation_direction_losses.append(val_direction_loss)
#             validation_vol_losses.append(val_vol_loss)
#             validation_trend_losses.append(val_trend_loss)
#             validation_epochs.append(epoch)
#             print(
#                 f"Validation Loss at Epoch {epoch}: {val_loss:.4f} "
#                 f"(abs={val_abs_loss:.4f}, dir={val_direction_loss:.4f}, vol={val_vol_loss:.4f}, trend={val_trend_loss:.4f})"
#             )
#             # save the model if it's the best so far
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 torch.save(model.state_dict(), "best_transformer_model.pth")
#                 print(f"New best model saved at epoch {epoch} with validation loss {best_val_loss:.4f}")

# # test the best model on the test set
# best_model = USDCNH_Transformer(input_size=MODEL_INPUT_SIZE, seq_len=SEQUENCE_LENGTH, nhead=2, num_layers=2, d_model=32, dropout=0.2).to(device)
# best_model.load_state_dict(torch.load("best_transformer_model.pth"))
# best_model.eval()
# with torch.no_grad():
#     test_error_history = [0.0] * ERROR_HISTORY_LEN
#     test_abs_chunks = []
#     test_direction_chunks = []
#     test_vol_chunks = []
#     test_trend_chunks = []

#     total_test_abs = 0.0
#     total_test_direction = 0.0
#     total_test_vol = 0.0
#     total_test_trend = 0.0
#     total_test_count = 0

#     for batch_X_test, batch_y_test in test_loader:
#         batch_X_test_with_error = append_error_feature(batch_X_test, test_error_history)
#         test_abs_batch, test_direction_batch, test_vol_pred_batch, test_trend_pred_batch = best_model(batch_X_test_with_error)
#         test_abs_chunks.append(test_abs_batch)
#         test_direction_chunks.append(test_direction_batch)
#         test_vol_chunks.append(test_vol_pred_batch)
#         test_trend_chunks.append(test_trend_pred_batch)

#         # Magnitude loss
#         test_abs_target = torch.abs(batch_y_test)
#         batch_abs = abs_loss_function(torch.relu(test_abs_batch), test_abs_target)
        
#         # Direction loss
#         batch_direction = loss_function(test_direction_batch, (batch_y_test > 0.5).float())
        
#         # Volatility loss
#         test_direction_prob = torch.sigmoid(test_direction_batch)
#         batch_realized_err = batch_y_test - test_direction_prob
#         batch_vol_target = torch.abs(batch_realized_err)
#         batch_vol = volatility_loss_function(torch.abs(test_vol_pred_batch), batch_vol_target)
        
#         # Trend loss
#         batch_trend_target = compute_trend_strength(batch_X_test)
#         batch_trend = trend_loss_function(torch.relu(test_trend_pred_batch), batch_trend_target)
        
#         bs = batch_X_test.shape[0]
#         total_test_abs += batch_abs.item() * bs
#         total_test_direction += batch_direction.item() * bs
#         total_test_vol += batch_vol.item() * bs
#         total_test_trend += batch_trend.item() * bs
#         total_test_count += bs

#         last_prob = torch.sigmoid(test_direction_batch[-1]).item()
#         last_binary_target = (batch_y_test[-1] > 0.5).float().item()  # Binarize for consistency
#         newest_error = last_binary_target - last_prob
#         test_error_history = update_error_history(test_error_history, newest_error)

#     test_abs_predictions = torch.cat(test_abs_chunks, dim=0)
#     test_direction_predictions = torch.cat(test_direction_chunks, dim=0)
#     test_vol_predictions = torch.cat(test_vol_chunks, dim=0)
#     test_trend_predictions = torch.cat(test_trend_chunks, dim=0)
    
#     test_direction_prob = torch.sigmoid(test_direction_predictions)
#     test_volatility = torch.abs(test_vol_predictions).mean().item()
#     test_trend_strength = torch.relu(test_trend_predictions).mean().item()
    
#     test_abs_loss = total_test_abs / total_test_count
#     test_direction_loss = total_test_direction / total_test_count
#     test_vol_loss = total_test_vol / total_test_count
#     test_trend_loss = total_test_trend / total_test_count
#     test_loss = test_direction_loss + ABS_LOSS_WEIGHT * test_abs_loss + VOLATILITY_LOSS_WEIGHT * test_vol_loss + TREND_LOSS_WEIGHT * test_trend_loss
    
#     # Compute combined confidence from all three signals
#     test_confidence = compute_combined_confidence(torch.relu(test_abs_predictions), test_direction_prob, test_vol_predictions)

#     # Base binary predictions come from direction head only.
#     # Confidence is used for filtering decisions, not rescaling probabilities.
#     binary_predictions = (test_direction_prob >= 0.5).float()
#     target_binary = (y_test_tensor > 0.5).float()
#     correct = (binary_predictions == target_binary).sum().item()
#     test_accuracy = correct / len(y_test_tensor)
#     test_confidence_mean = test_confidence.mean().item()

#     # Confidence-filtered evaluation: report both coverage and accuracy.
#     mask_55 = test_confidence >= 0.55
#     mask_60 = test_confidence >= 0.60
#     mask_70 = test_confidence >= 0.70

#     cov_55 = mask_55.float().mean().item()
#     cov_60 = mask_60.float().mean().item()
#     cov_70 = mask_70.float().mean().item()

#     if mask_55.any():
#         acc_55 = (binary_predictions[mask_55] == target_binary[mask_55]).float().mean().item()
#     else:
#         acc_55 = float("nan")
#     if mask_60.any():
#         acc_60 = (binary_predictions[mask_60] == target_binary[mask_60]).float().mean().item()
#     else:
#         acc_60 = float("nan")
#     if mask_70.any():
#         acc_70 = (binary_predictions[mask_70] == target_binary[mask_70]).float().mean().item()
#     else:
#         acc_70 = float("nan")

#     test_accuracy_conf_55 = acc_55
#     test_coverage_conf_55 = cov_55
#     test_accuracy_conf_60 = acc_60
#     test_coverage_conf_60 = cov_60
#     test_accuracy_conf_70 = acc_70
#     test_coverage_conf_70 = cov_70
#     print(
#         f"Test Loss: {test_loss:.4f} (abs={test_abs_loss:.4f}, dir={test_direction_loss:.4f}, vol={test_vol_loss:.4f}, trend={test_trend_loss:.4f}), "
#         f"Test Accuracy: {test_accuracy:.4f}, Mean Confidence: {test_confidence_mean:.4f}"
#     )
#     print(
#         f"Auxiliary Metrics: Mean Volatility: {test_volatility:.4f}, Mean Trend Strength: {test_trend_strength:.4f}"
#     )
#     print(
#         f"Confidence Filtered Metrics: "
#         f"conf>=0.55 acc={acc_55:.4f}, cov={cov_55:.4f} | "
#         f"conf>=0.60 acc={acc_60:.4f}, cov={cov_60:.4f} | "
#         f"conf>=0.70 acc={acc_70:.4f}, cov={cov_70:.4f}"
#     )

# # ==========================================
# # SAVE METRICS AND CREATE VISUALIZATION
# # ==========================================
# # Create a metrics dictionary
# metrics = {
#     'training_losses': training_losses,
#     'training_abs_losses': training_abs_losses,
#     'training_direction_losses': training_direction_losses,
#     'training_vol_losses': training_vol_losses,
#     'training_trend_losses': training_trend_losses,
#     'validation_losses': validation_losses,
#     'validation_abs_losses': validation_abs_losses,
#     'validation_direction_losses': validation_direction_losses,
#     'validation_vol_losses': validation_vol_losses,
#     'validation_trend_losses': validation_trend_losses,
#     'validation_epochs': validation_epochs,
#     'test_accuracy': test_accuracy,
#     'test_confidence_mean': test_confidence_mean,
#     'test_accuracy_conf_55': test_accuracy_conf_55,
#     'test_coverage_conf_55': test_coverage_conf_55,
#     'test_accuracy_conf_60': test_accuracy_conf_60,
#     'test_coverage_conf_60': test_coverage_conf_60,
#     'test_accuracy_conf_70': test_accuracy_conf_70,
#     'test_coverage_conf_70': test_coverage_conf_70,
#     'test_volatility': test_volatility,
#     'test_trend_strength': test_trend_strength
# }

# # Save training metrics to CSV with all components
# metrics_df = pd.DataFrame({
#     'epoch': list(range(1, len(training_losses) + 1)),
#     'total_loss': training_losses,
#     'abs_loss': training_abs_losses,
#     'direction_loss': training_direction_losses,
#     'vol_loss': training_vol_losses,
#     'trend_loss': training_trend_losses
# })
# metrics_df.to_csv('training_metrics.csv', index=False)

# # Create validation metrics CSV with all components
# val_metrics_df = pd.DataFrame({
#     'epoch': validation_epochs,
#     'total_loss': validation_losses,
#     'abs_loss': validation_abs_losses,
#     'direction_loss': validation_direction_losses,
#     'vol_loss': validation_vol_losses,
#     'trend_loss': validation_trend_losses
# })
# val_metrics_df.to_csv('validation_metrics.csv', index=False)

# # Create comprehensive plot
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# # Plot 1: Training and Validation Loss
# axes[0].plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss', linewidth=2, alpha=0.8)
# axes[0].plot(validation_epochs, validation_losses, label='Validation Loss', marker='o', linewidth=2, markersize=5, alpha=0.8)
# axes[0].set_xlabel('Epoch', fontsize=12)
# axes[0].set_ylabel('Loss', fontsize=12)
# axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
# axes[0].legend(fontsize=11)
# axes[0].grid(True, alpha=0.3)

# # Plot 2: Test Accuracy (as a reference)
# axes[1].text(0.5, 0.5, f'Test Accuracy: {test_accuracy:.4f}\nMean Confidence: {test_confidence_mean:.4f}\n\nVolatility: {test_volatility:.4f}\nTrend Strength: {test_trend_strength:.4f}\n\nBest Validation Loss: {min(validation_losses):.4f}',
#              ha='center', va='center', fontsize=14, fontweight='bold', 
#              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=1))
# axes[1].axis('off')
# axes[1].set_title('Test Results', fontsize=14, fontweight='bold')

# plt.tight_layout()
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# plot_filename = f'training_plot_{timestamp}.png'
# plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
# print(f"\n✓ Plot saved to: {plot_filename}")
# print(f"✓ Training metrics saved to: training_metrics.csv")
# print(f"✓ Validation metrics saved to: validation_metrics.csv")
# plt.show()






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
            train_error_history = [0.0] * ERROR_HISTORY_LEN
            
            # ⭐️ Iterate through the mini-batches!
            for batch_X, batch_y in train_loader:
                agent.optimizer.zero_grad()
                
                # Apply Data Augmentation strictly to this tiny batch
                batch_X_aug = augment_financial_data(batch_X, noise_std=0.05, mask_prob=0.10)
                batch_X_with_error = append_error_feature(batch_X_aug, train_error_history)
                
                # Forward pass on the batch
                abs_logits, direction_logits, volatility_pred, trend_pred = agent.active_model(batch_X_with_error)

                # Multi-head losses
                abs_target = torch.abs(batch_y)
                abs_loss = abs_loss_function(torch.relu(abs_logits), abs_target)

                direction_target = (batch_y > 0.5).float()
                direction_loss = loss_function(direction_logits, direction_target)

                direction_prob = torch.sigmoid(direction_logits)
                realized_error = batch_y - direction_prob
                volatility_target = torch.abs(realized_error)
                vol_loss = volatility_loss_function(torch.abs(volatility_pred), volatility_target)

                trend_target = compute_trend_strength(batch_X)
                trend_loss = trend_loss_function(torch.relu(trend_pred), trend_target)

                loss = direction_loss + ABS_LOSS_WEIGHT * abs_loss + VOLATILITY_LOSS_WEIGHT * vol_loss + TREND_LOSS_WEIGHT * trend_loss
                
                # Backprop
                loss.backward()
                
                # Gradient Clipping (Crucial for Transformers)
                torch.nn.utils.clip_grad_norm_(agent.active_model.parameters(), max_norm=1.0)
                
                # Update weights
                agent.optimizer.step()

                # Update online error history for autoregressive feature
                with torch.no_grad():
                    last_prob = torch.sigmoid(direction_logits[-1]).item()
                    last_binary_target = (batch_y[-1] > 0.5).float().item()
                    newest_error = last_binary_target - last_prob
                    train_error_history = update_error_history(train_error_history, newest_error)
            
            # ⭐️ Update EMA Shadow Model at the end of every epoch
            agent.update_ema(decay=0.80)
            
            # Update the inner bar with the final batch loss of the epoch
            agent_pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Dir': f"{direction_loss.item():.4f}"})

    # ------------------------------------------
    # STEP B: EVALUATION (Mini-Batched)
    # ------------------------------------------
    for agent in population:
        agent.ema_model.eval()
        
        total_val_loss = 0.0
        total_val_count = 0
        
        with torch.no_grad():
            val_error_history = [0.0] * ERROR_HISTORY_LEN
            # ⭐️ Iterate through the Validation DataLoader safely!
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val_with_error = append_error_feature(batch_X_val, val_error_history)
                val_abs, val_direction, val_vol, val_trend = agent.ema_model(batch_X_val_with_error)

                val_abs_target = torch.abs(batch_y_val)
                val_abs_loss = abs_loss_function(torch.relu(val_abs), val_abs_target)

                val_direction_target = (batch_y_val > 0.5).float()
                val_direction_loss = loss_function(val_direction, val_direction_target)

                val_direction_prob = torch.sigmoid(val_direction)
                val_realized_error = batch_y_val - val_direction_prob
                val_vol_target = torch.abs(val_realized_error)
                val_vol_loss = volatility_loss_function(torch.abs(val_vol), val_vol_target)

                val_trend_target = compute_trend_strength(batch_X_val)
                val_trend_loss = trend_loss_function(torch.relu(val_trend), val_trend_target)

                batch_loss = val_direction_loss + ABS_LOSS_WEIGHT * val_abs_loss + VOLATILITY_LOSS_WEIGHT * val_vol_loss + TREND_LOSS_WEIGHT * val_trend_loss

                bs = batch_X_val.shape[0]
                total_val_loss += batch_loss.item() * bs
                total_val_count += bs

                last_prob = torch.sigmoid(val_direction[-1]).item()
                last_binary_target = (batch_y_val[-1] > 0.5).float().item()
                newest_error = last_binary_target - last_prob
                val_error_history = update_error_history(val_error_history, newest_error)
        
        # ⭐️ Calculate the true average loss across the entire validation set
        avg_val_loss = total_val_loss / total_val_count
        
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
    test_error_history = [0.0] * ERROR_HISTORY_LEN
    test_abs_chunks = []
    test_direction_chunks = []
    test_vol_chunks = []

    total_test_abs = 0.0
    total_test_direction = 0.0
    total_test_vol = 0.0
    total_test_trend = 0.0
    total_test_count = 0

    for batch_X_test, batch_y_test in test_loader:
        batch_X_test_with_error = append_error_feature(batch_X_test, test_error_history)
        test_abs, test_direction, test_vol, test_trend = ultimate_model(batch_X_test_with_error)

        test_abs_chunks.append(test_abs)
        test_direction_chunks.append(test_direction)
        test_vol_chunks.append(test_vol)

        batch_abs_target = torch.abs(batch_y_test)
        batch_abs_loss = abs_loss_function(torch.relu(test_abs), batch_abs_target)

        batch_direction_target = (batch_y_test > 0.5).float()
        batch_direction_loss = loss_function(test_direction, batch_direction_target)

        batch_direction_prob = torch.sigmoid(test_direction)
        batch_realized_error = batch_y_test - batch_direction_prob
        batch_vol_target = torch.abs(batch_realized_error)
        batch_vol_loss = volatility_loss_function(torch.abs(test_vol), batch_vol_target)

        batch_trend_target = compute_trend_strength(batch_X_test)
        batch_trend_loss = trend_loss_function(torch.relu(test_trend), batch_trend_target)

        bs = batch_X_test.shape[0]
        total_test_abs += batch_abs_loss.item() * bs
        total_test_direction += batch_direction_loss.item() * bs
        total_test_vol += batch_vol_loss.item() * bs
        total_test_trend += batch_trend_loss.item() * bs
        total_test_count += bs

        last_prob = torch.sigmoid(test_direction[-1]).item()
        last_binary_target = (batch_y_test[-1] > 0.5).float().item()
        newest_error = last_binary_target - last_prob
        test_error_history = update_error_history(test_error_history, newest_error)

    test_abs_predictions = torch.cat(test_abs_chunks, dim=0)
    test_direction_predictions = torch.cat(test_direction_chunks, dim=0)
    test_vol_predictions = torch.cat(test_vol_chunks, dim=0)

    test_direction_prob = torch.sigmoid(test_direction_predictions)
    test_confidence = compute_combined_confidence(torch.relu(test_abs_predictions), test_direction_prob, test_vol_predictions)
    binary_predictions = (test_direction_prob >= 0.5).float()
    target_binary = (y_test_tensor > 0.5).float()

    correct = (binary_predictions == target_binary).sum().item()
    accuracy = correct / len(y_test_tensor)

    test_abs_loss = total_test_abs / total_test_count
    test_direction_loss = total_test_direction / total_test_count
    test_vol_loss = total_test_vol / total_test_count
    test_trend_loss = total_test_trend / total_test_count
    test_loss = test_direction_loss + ABS_LOSS_WEIGHT * test_abs_loss + VOLATILITY_LOSS_WEIGHT * test_vol_loss + TREND_LOSS_WEIGHT * test_trend_loss

    mask_60 = test_confidence >= 0.60
    cov_60 = mask_60.float().mean().item()
    if mask_60.any():
        acc_60 = (binary_predictions[mask_60] == target_binary[mask_60]).float().mean().item()
    else:
        acc_60 = float("nan")

print(f"Ultimate Test Loss: {test_loss:.4f} (abs={test_abs_loss:.4f}, dir={test_direction_loss:.4f}, vol={test_vol_loss:.4f}, trend={test_trend_loss:.4f})")
print(f"Ultimate Test Accuracy:   {accuracy * 100:.2f}%")
print(f"Ultimate Mean Confidence: {test_confidence.mean().item():.4f}")
print(f"Ultimate conf>=0.60: acc={acc_60:.4f}, cov={cov_60:.4f}")

torch.save(ultimate_model.state_dict(), "USDCNH_PBT_EMA_Production.pth")
print("\nMaster Model saved to 'USDCNH_PBT_EMA_Production.pth'.")