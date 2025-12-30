import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Path to your Dataset
DATASET_ROOT = r"D:\anechoic_dataset_v2"

# Hyperparameters
BATCH_SIZE = 128         # 128 consumes around 3.7G/4G for a RTX3050 laptop GPU
LEARNING_RATE = 1e-3     # Standard Adam LR
N_EPOCHS = 60            # Max number of epochs
PATIENCE = 10            # Early Stopping Patience
N_FFT = 512              # Number of FFT bins
HOP_LENGTH = 160         # 10ms at 16kHz
HIDDEN_SIZE = 320        # Tuned for the target parameter size
NUM_WORKERS = 4          # Number of CPU threads for data loading

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. DATASET LOADER
# ==========================================
class AnechoicDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, max_len_seconds=4.0):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * max_len_seconds)
        
        # Fast scanning of sample directories
        print(f"Scanning dataset at {root_dir}...")
        self.sample_folders = sorted(glob.glob(os.path.join(root_dir, "sample_*")))
        
        if len(self.sample_folders) == 0:
            raise ValueError(f"No sample folders found in {root_dir}!")
            
        print(f"Found {len(self.sample_folders)} samples.")

    def __len__(self):
        return len(self.sample_folders)

    def __getitem__(self, idx):
        folder = self.sample_folders[idx]
        mix_path = os.path.join(folder, "mixture.wav")
        target_path = os.path.join(folder, "target.wav")
        
        # Load Audio [Channels, Time]
        mix_wav, sr = torchaudio.load(mix_path)
        target_wav, _ = torchaudio.load(target_path)
        
        # Safety Check: Length
        if mix_wav.shape[1] > self.max_len:
            mix_wav = mix_wav[:, :self.max_len]
            target_wav = target_wav[:, :self.max_len]
        
        return mix_wav, target_wav


# ==========================================
# 3. LSTM MODEL 
# ==========================================
class StatefulStereoMaskNet(nn.Module):
    def __init__(self, n_fft=512, hidden_size=320, num_layers=2):
        super(StatefulStereoMaskNet, self).__init__()
        self.n_fft = n_fft
        self.freq_bins = n_fft // 2 + 1
        
        # Input: LogMag + CosIPD + SinIPD = 3 features per freq bin
        input_dim = self.freq_bins * 3 
        
        # Uni-directional LSTM for live streaming
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            bidirectional=False)
        
        self.fc = nn.Linear(hidden_size, self.freq_bins)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_left_stft, x_right_stft):
        # 1. Feature Extraction
        mag_left = torch.abs(x_left_stft)
        log_mag = torch.log1p(mag_left)
        
        phase_left = torch.angle(x_left_stft)
        phase_right = torch.angle(x_right_stft)
        ipd = phase_left - phase_right
        cos_ipd = torch.cos(ipd)
        sin_ipd = torch.sin(ipd)
        
        # Stack Features & Permute for LSTM: [Batch, Time, Features]
        x = torch.cat([log_mag, cos_ipd, sin_ipd], dim=1)
        x = x.permute(0, 2, 1) 
        
        # 2. LSTM
        lstm_out, _ = self.lstm(x)
        
        # 3. Mask
        mask = self.sigmoid(self.fc(lstm_out)) # [Batch, Time, Freq]
        mask = mask.permute(0, 2, 1)           # [Batch, Freq, Time]
        
        # 4. Apply to Reference (Left Channel)
        separated_stft = x_left_stft * mask
        
        return separated_stft, mask


# ==========================================
# 4. LOSS FUNCTION (SI-SDR + STOI)
# ==========================================
class SISDR_STOI_Loss(nn.Module):
    def __init__(self, n_fft=512, hop_length=160, alpha_sisdr=1.0, alpha_stoi=5.0):
        super(SISDR_STOI_Loss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Register window as a buffer so it moves to GPU automatically
        self.register_buffer('window', torch.hann_window(n_fft))
        self.alpha_sisdr = alpha_sisdr
        self.alpha_stoi = alpha_stoi

    def si_sdr_loss(self, estimate, reference):
        eps = 1e-8
        # Ensure zero mean
        estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
        reference = reference - torch.mean(reference, dim=-1, keepdim=True)
        
        # Calculate optimal scaling (Alpha)
        ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True) + eps
        scaling = torch.sum(estimate * reference, dim=-1, keepdim=True) / ref_energy
        
        # Project estimate onto reference
        target_projection = scaling * reference
        noise = estimate - target_projection
        
        # SI-SDR ratio
        ratio = torch.sum(target_projection ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
        si_sdr = 10 * torch.log10(ratio + eps)
        
        return -torch.mean(si_sdr)

    def stoi_proxy_loss(self, est_stft, ref_stft):
        # Differentiable Proxy for STOI
        eps = 1e-8
        mag_est = torch.abs(est_stft) + eps
        mag_ref = torch.abs(ref_stft) + eps
        
        # Normalize across Frequency dimension
        mean_est = torch.mean(mag_est, dim=1, keepdim=True)
        std_est = torch.std(mag_est, dim=1, keepdim=True) + eps
        norm_est = (mag_est - mean_est) / std_est
        
        mean_ref = torch.mean(mag_ref, dim=1, keepdim=True)
        std_ref = torch.std(mag_ref, dim=1, keepdim=True) + eps
        norm_ref = (mag_ref - mean_ref) / std_ref
        
        # Cosine similarity
        correlation = torch.mean(norm_est * norm_ref)
        return -correlation

    def forward(self, predicted_stft, target_waveform):
        # 1. Reconstruct Waveform using ISTFT
        predicted_waveform = torch.istft(
            predicted_stft, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.window, 
            return_complex=False
        )
        
        # 2. Match Lengths
        min_len = min(predicted_waveform.shape[-1], target_waveform.shape[-1])
        predicted_waveform = predicted_waveform[..., :min_len]
        target_waveform = target_waveform[..., :min_len]
        
        # 3. Calculate SI-SDR
        loss_sisdr = self.si_sdr_loss(predicted_waveform, target_waveform)
        
        # 4. Calculate STOI
        with torch.no_grad():
            target_stft = torch.stft(
                target_waveform, 
                self.n_fft, 
                self.hop_length, 
                window=self.window, 
                return_complex=True
            )
            # Align STFT shapes
            if target_stft.shape[-1] != predicted_stft.shape[-1]:
                 min_frames = min(target_stft.shape[-1], predicted_stft.shape[-1])
                 target_stft = target_stft[..., :min_frames]
                 predicted_stft_trunc = predicted_stft[..., :min_frames]
            else:
                 predicted_stft_trunc = predicted_stft
        
        loss_stoi = self.stoi_proxy_loss(predicted_stft_trunc, target_stft)
        
        # Weighted Sum
        total_loss = (self.alpha_sisdr * loss_sisdr) + (self.alpha_stoi * loss_stoi)
        return total_loss, loss_sisdr.item(), loss_stoi.item()


# ==========================================
# 5. TRAINING LOOP
# ==========================================
def main():
    print(f"--- Training on {DEVICE} ---")
    # --- Data ---
    full_dataset = AnechoicDataset(DATASET_ROOT)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # --- Model ---
    model = StatefulStereoMaskNet(n_fft=N_FFT, hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Loss Functions
    criterion_warmup = nn.L1Loss()
    criterion_main = SISDR_STOI_Loss(n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEVICE)
    
    # STFT Helper
    torch_window = torch.hann_window(N_FFT).to(DEVICE)
    
    best_val_loss = float('inf')
    patience_counter = 0 
    
    print(f"Starting Training: {len(train_ds)} train, {len(val_ds)} validation samples.")
    print(f"Early Stopping Patience: {PATIENCE} epochs")
    
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss_total = 0
        
        use_warmup = (epoch < 1) 
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        
        for batch_idx, (mix_wav, target_wav) in enumerate(loop):
            mix_wav = mix_wav.to(DEVICE)
            target_wav = target_wav.to(DEVICE).squeeze(1)
            
            # Compute STFT
            B, C, T = mix_wav.shape
            mix_flat = mix_wav.view(B*C, T)
            stft_mix = torch.stft(mix_flat, N_FFT, HOP_LENGTH, window=torch_window, return_complex=True)
            stft_mix = stft_mix.view(B, C, stft_mix.shape[1], stft_mix.shape[2])
            
            stft_left = stft_mix[:, 0, :, :]
            stft_right = stft_mix[:, 1, :, :]
            
            # Forward
            predicted_stft, mask = model(stft_left, stft_right)
            
            # Loss
            if use_warmup:
                with torch.no_grad():
                    target_stft = torch.stft(target_wav, N_FFT, HOP_LENGTH, window=torch_window, return_complex=True)
                loss = criterion_warmup(torch.abs(predicted_stft), torch.abs(target_stft))
                stats = "Warmup L1"
            else:
                loss, sisdr, stoi = criterion_main(predicted_stft, target_wav)
                stats = f"SI-SDR: {sisdr:.2f} | STOI: {stoi:.2f}"
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss_total += loss.item()
            loop.set_postfix(loss=loss.item(), details=stats)
            
        # --- Validation ---
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for mix_wav, target_wav in val_loader:
                mix_wav, target_wav = mix_wav.to(DEVICE), target_wav.to(DEVICE).squeeze(1)
                
                B, C, T = mix_wav.shape
                stft_mix = torch.stft(mix_wav.view(B*C, T), N_FFT, HOP_LENGTH, window=torch_window, return_complex=True)
                stft_mix = stft_mix.view(B, C, stft_mix.shape[1], stft_mix.shape[2])
                
                pred_stft, _ = model(stft_mix[:, 0], stft_mix[:, 1])
                
                if use_warmup:
                    target_stft = torch.stft(target_wav, N_FFT, HOP_LENGTH, window=torch_window, return_complex=True)
                    loss = criterion_warmup(torch.abs(pred_stft), torch.abs(target_stft))
                else:
                    loss, _, _ = criterion_main(pred_stft, target_wav)
                
                val_loss_total += loss.item()
        
        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # --- EARLY STOPPING CHECK ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset counter
            torch.save(model.state_dict(), "LSTM_model.pth")
            print(">>> New Best Model Saved!")
        else:
            patience_counter += 1
            print(f"No improvement. Early Stopping Counter: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print("!!! Early stopping triggered. Training stopped. !!!")
                break
            
    print("Training Complete.")

if __name__ == "__main__":
    main()