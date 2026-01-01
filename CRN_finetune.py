import os
import glob
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION (FINE-TUNING SETUP)
# ==========================================
# CHANGE THIS to the folder containing your specific Male/Female data
DATASET_ROOT = r"D:\anechoic_dataset_v3_male_female" 

# PATH to your existing 60-epoch model
PRETRAINED_PATH = "CRN_Model.pth"

# OUTPUT PATH for the new model
NEW_MODEL_NAME = "CRN_Model_FineTuned.pth"

# HYPERPARAMETERS
BATCH_SIZE = 128          # Adjust based on your VRAM (32 or 64 is often good)
LEARNING_RATE = 1e-4     # LOWER LR for Fine-Tuning (was 1e-3)
N_EPOCHS = 30            # Fewer epochs needed for fine-tuning
PATIENCE = 5
N_FFT = 512
HOP_LENGTH = 160
SILENCE_PROB = 0.1       # Lower silence prob to focus on active separation
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DATASET LOADER
# ==========================================
class RoomAcousticDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, fixed_length=3.0, silence_prob=0.1):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * fixed_length)
        self.silence_prob = silence_prob
        
        print(f"Scanning dataset at {root_dir}...")
        self.sample_folders = sorted(glob.glob(os.path.join(root_dir, "sample_*")))
        
        if len(self.sample_folders) == 0:
            print(f"WARNING: No 'sample_XXXXX' folders found in {root_dir}")
            print("Please check DATASET_ROOT path.")
        else:
            print(f"Found {len(self.sample_folders)} samples. Silence Prob: {silence_prob}")

    def __len__(self):
        return len(self.sample_folders)

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        if waveform.shape[-1] < self.num_samples:
            pad_amt = self.num_samples - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad_amt))
        else:
            waveform = waveform[:, :self.num_samples]
        return waveform

    def __getitem__(self, idx):
        folder_path = self.sample_folders[idx]
        mix_path = os.path.join(folder_path, "mixture.wav")
        target_path = os.path.join(folder_path, "target.wav")
        meta_path = os.path.join(folder_path, "meta.json")
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            target_angle = float(meta['target_angle'])
            interf_angle = float(meta['interf_angle'])
            
        mixture = self._load_audio(mix_path)
        
        # Negative Sampling (Silence Training)
        if random.random() < self.silence_prob:
            valid_angle = False
            while not valid_angle:
                random_angle = random.uniform(0, 180)
                if abs(random_angle - target_angle) > 20 and abs(random_angle - interf_angle) > 20:
                    input_angle = random_angle
                    valid_angle = True
            ground_truth = torch.zeros(1, self.num_samples)
        else:
            input_angle = target_angle
            ground_truth = self._load_audio(target_path)
            if ground_truth.shape[0] > 1:
                ground_truth = ground_truth[0:1, :] # Force Mono

        return mixture, torch.tensor([input_angle], dtype=torch.float32), ground_truth

# ==========================================
# 3. SPATIAL MEDIUM-CRN MODEL
# ==========================================
class SpatialMediumCRN(nn.Module):
    def __init__(self, n_fft=512, hop_length=160):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))
        
        # Encoder (2ch -> 256ch)
        self.enc1 = nn.Conv2d(4, 32, (3,3), stride=(2,1), padding=(1,1))
        self.enc2 = nn.Conv2d(32, 64, (3,3), stride=(2,1), padding=(1,1))
        self.enc3 = nn.Conv2d(64, 128, (3,3), stride=(2,1), padding=(1,1))
        self.enc4 = nn.Conv2d(128, 256, (3,3), stride=(2,1), padding=(1,1))
        
        # Angle Net
        self.angle_net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 256)
        )
        
        # Bottleneck
        self.gru_input_dim = 256 * 17
        self.gru_hidden_dim = 256
        self.gru = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, batch_first=True)
        self.gru_fc = nn.Linear(self.gru_hidden_dim, self.gru_input_dim)
        
        # Decoder
        self.dec4 = nn.ConvTranspose2d(512, 128, (3,3), stride=(2,1), padding=(1,1), output_padding=(0,0))
        self.dec3 = nn.ConvTranspose2d(256, 64, (3,3), stride=(2,1), padding=(1,1), output_padding=(0,0))
        self.dec2 = nn.ConvTranspose2d(128, 32, (3,3), stride=(2,1), padding=(1,1), output_padding=(0,0))
        self.dec1 = nn.ConvTranspose2d(64, 2, (3,3), stride=(2,1), padding=(1,1), output_padding=(0,0))

    def forward(self, x, angle):
        # STFT
        stft = torch.stft(x.reshape(-1, x.shape[-1]), self.n_fft, self.hop_length, window=self.window, return_complex=True)
        stft = stft.view(x.shape[0], 2, stft.shape[-2], stft.shape[-1])
        feat = torch.cat([stft.real, stft.imag], dim=1)
        
        # Encode
        e1 = F.elu(self.enc1(feat))
        e2 = F.elu(self.enc2(e1))
        e3 = F.elu(self.enc3(e2))
        e4 = F.elu(self.enc4(e3))
        
        # Angle Injection
        rad = torch.deg2rad(angle)
        angle_vec = torch.cat([torch.sin(rad), torch.cos(rad)], dim=1)
        angle_emb = self.angle_net(angle_vec).unsqueeze(-1).unsqueeze(-1)
        
        # Bottleneck
        b, c, f, t = e4.shape
        gru_in = e4.permute(0, 3, 1, 2).reshape(b, t, -1)
        gru_out, _ = self.gru(gru_in)
        gru_out = F.relu(self.gru_fc(gru_out)).reshape(b, t, c, f).permute(0, 2, 3, 1)
        
        # Gate
        gru_out = gru_out + angle_emb
        
        # Decode
        d4 = F.elu(self.dec4(torch.cat([gru_out, e4], dim=1)))
        d3 = F.elu(self.dec3(torch.cat([d4, e3], dim=1)))
        d2 = F.elu(self.dec2(torch.cat([d3, e2], dim=1)))
        mask = self.dec1(torch.cat([d2, e1], dim=1))
        
        # Apply Mask
        ref = stft[:, 0]
        m_real, m_imag = mask[:, 0], mask[:, 1]
        est_real = ref.real * m_real - ref.imag * m_imag
        est_imag = ref.real * m_imag + ref.imag * m_real
        est_complex = torch.complex(est_real, est_imag)
        
        return torch.istft(est_complex, self.n_fft, self.hop_length, window=self.window)

# ==========================================
# 4. LOSS FUNCTION
# ==========================================
class SpatialSeparationLoss(nn.Module):
    def __init__(self, n_fft=512, hop_length=160, alpha_sisdr=1.0, alpha_spectral=5.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha_sisdr = alpha_sisdr
        self.alpha_spectral = alpha_spectral
        self.register_buffer('window', torch.hann_window(n_fft))
        self.mse = nn.MSELoss()

    def si_sdr(self, estimate, reference):
        eps = 1e-8
        est = estimate - torch.mean(estimate, dim=-1, keepdim=True)
        ref = reference - torch.mean(reference, dim=-1, keepdim=True)
        ref_energy = torch.sum(ref ** 2, dim=-1, keepdim=True) + eps
        projection = torch.sum(est * ref, dim=-1, keepdim=True) * ref / ref_energy
        noise = est - projection
        ratio = torch.sum(projection ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
        return -10 * torch.log10(ratio + eps).mean()

    def spectral_correlation(self, estimate, reference):
        est_stft = torch.stft(estimate, self.n_fft, self.hop_length, window=self.window, return_complex=True)
        ref_stft = torch.stft(reference, self.n_fft, self.hop_length, window=self.window, return_complex=True)
        mag_est = torch.abs(est_stft) + 1e-8
        mag_ref = torch.abs(ref_stft) + 1e-8
        
        mu_est = torch.mean(mag_est, dim=1, keepdim=True)
        std_est = torch.std(mag_est, dim=1, keepdim=True) + 1e-8
        norm_est = (mag_est - mu_est) / std_est
        
        mu_ref = torch.mean(mag_ref, dim=1, keepdim=True)
        std_ref = torch.std(mag_ref, dim=1, keepdim=True) + 1e-8
        norm_ref = (mag_ref - mu_ref) / std_ref
        return -torch.mean(norm_est * norm_ref)

    def forward(self, estimate, reference):
        min_len = min(estimate.shape[-1], reference.shape[-1])
        estimate = estimate[..., :min_len]
        reference = reference[..., :min_len]

        ref_energy = torch.sum(reference ** 2, dim=-1)
        has_speech = ref_energy > 1e-5
        
        total_loss = torch.tensor(0.0, device=estimate.device)
        count = 0

        if has_speech.any():
            l_time = self.si_sdr(estimate[has_speech], reference[has_speech])
            l_freq = self.spectral_correlation(estimate[has_speech], reference[has_speech])
            total_loss += (self.alpha_sisdr * l_time) + (self.alpha_spectral * l_freq)
            count += 1
            
        if (~has_speech).any():
            l_silence = self.mse(estimate[~has_speech], reference[~has_speech]) * 1000.0 
            total_loss += l_silence
            count += 1

        return total_loss / max(count, 1)

# ==========================================
# 5. FINE-TUNING LOOP
# ==========================================
def main():
    print(f"--- Fine-Tuning Spatial Audio Model on {DEVICE} ---")
    
    # 1. Dataset Setup
    try:
        full_dataset = RoomAcousticDataset(DATASET_ROOT, silence_prob=SILENCE_PROB)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # 2. Model Initialization
    model = SpatialMediumCRN(n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEVICE)
    
    # ========================================================
    # LOAD PRE-TRAINED WEIGHTS
    # ========================================================
    if os.path.exists(PRETRAINED_PATH):
        print(f"\n[INFO] Loading pre-trained weights from: {PRETRAINED_PATH}")
        try:
            state_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
            print("[INFO] Weights loaded successfully. Proceeding to fine-tune.")
        except Exception as e:
            print(f"[ERROR] Failed to load weights: {e}")
            return
    else:
        print(f"\n[WARNING] Pre-trained file '{PRETRAINED_PATH}' NOT FOUND!")
        print("Training will start from scratch (Random Initialization).")
        input("Press Enter to continue anyway, or Ctrl+C to stop...")

    # 3. Optimizer & Loss
    # We re-initialize the optimizer for fine-tuning to reset momentum/buffers
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = SpatialSeparationLoss().to(DEVICE)
    
    best_val_loss = float('inf')
    patience_counter = 0 
    
    print(f"\nStarting Fine-Tuning: {len(train_ds)} train, {len(val_ds)} validation.")
    print(f"Targeting: {NEW_MODEL_NAME}\n")
    
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss_total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        
        for mixture, angle, target in loop:
            mixture, angle, target = mixture.to(DEVICE), angle.to(DEVICE), target.to(DEVICE).squeeze(1)
            
            # Forward
            estimate = model(mixture, angle)
            loss = criterion(estimate, target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss_total += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for mixture, angle, target in val_loader:
                mixture, angle, target = mixture.to(DEVICE), angle.to(DEVICE), target.to(DEVICE).squeeze(1)
                estimate = model(mixture, angle)
                val_loss_total += criterion(estimate, target).item()
        
        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Save Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), NEW_MODEL_NAME)
            print(f">>> New Best Model Saved: {NEW_MODEL_NAME}")
        else:
            patience_counter += 1
            print(f"No improvement. Counter: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("!!! Early stopping triggered !!!")
                break
            
    print("Fine-Tuning Complete.")

if __name__ == "__main__":
    main()