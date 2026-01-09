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
# 1. CONFIGURATION (COMPRESSED PESQ OPTIMIZATION)
# ==========================================
DATASET_ROOT = r"D:\reverb_dataset_v3_male_female" 
PRETRAINED_PATH = "CRN_Reverb"
NEW_MODEL_NAME = "CRN_Reverb_finetuned.pth" 

# HYPERPARAMETERS
BATCH_SIZE = 128
LEARNING_RATE = 1e-3      
N_EPOCHS = 60             # Increased to 20 for Compressed Loss convergence
PATIENCE = 20              # Relaxed slightly to allow for fine detail learning
N_FFT = 512
HOP_LENGTH = 160
SILENCE_PROB = 0.3
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
        
        self.enc1 = nn.Conv2d(4, 32, (3,3), stride=(2,1), padding=(1,1))
        self.enc2 = nn.Conv2d(32, 64, (3,3), stride=(2,1), padding=(1,1))
        self.enc3 = nn.Conv2d(64, 128, (3,3), stride=(2,1), padding=(1,1))
        self.enc4 = nn.Conv2d(128, 256, (3,3), stride=(2,1), padding=(1,1))
        
        self.angle_net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 256)
        )
        
        self.gru_input_dim = 256 * 17
        self.gru_hidden_dim = 256
        self.gru = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, batch_first=True)
        self.gru_fc = nn.Linear(self.gru_hidden_dim, self.gru_input_dim)
        
        self.dec4 = nn.ConvTranspose2d(512, 128, (3,3), stride=(2,1), padding=(1,1), output_padding=(0,0))
        self.dec3 = nn.ConvTranspose2d(256, 64, (3,3), stride=(2,1), padding=(1,1), output_padding=(0,0))
        self.dec2 = nn.ConvTranspose2d(128, 32, (3,3), stride=(2,1), padding=(1,1), output_padding=(0,0))
        self.dec1 = nn.ConvTranspose2d(64, 2, (3,3), stride=(2,1), padding=(1,1), output_padding=(0,0))

    def forward(self, x, angle):
        stft = torch.stft(x.reshape(-1, x.shape[-1]), self.n_fft, self.hop_length, window=self.window, return_complex=True)
        stft = stft.view(x.shape[0], 2, stft.shape[-2], stft.shape[-1])
        feat = torch.cat([stft.real, stft.imag], dim=1)
        
        e1 = F.elu(self.enc1(feat))
        e2 = F.elu(self.enc2(e1))
        e3 = F.elu(self.enc3(e2))
        e4 = F.elu(self.enc4(e3))
        
        rad = torch.deg2rad(angle)
        angle_vec = torch.cat([torch.sin(rad), torch.cos(rad)], dim=1)
        angle_emb = self.angle_net(angle_vec).unsqueeze(-1).unsqueeze(-1)
        
        b, c, f, t = e4.shape
        gru_in = e4.permute(0, 3, 1, 2).reshape(b, t, -1)
        gru_out, _ = self.gru(gru_in)
        gru_out = F.relu(self.gru_fc(gru_out)).reshape(b, t, c, f).permute(0, 2, 3, 1)
        gru_out = gru_out + angle_emb
        
        d4 = F.elu(self.dec4(torch.cat([gru_out, e4], dim=1)))
        d3 = F.elu(self.dec3(torch.cat([d4, e3], dim=1)))
        d2 = F.elu(self.dec2(torch.cat([d3, e2], dim=1)))
        mask = self.dec1(torch.cat([d2, e1], dim=1))
        
        ref = stft[:, 0]
        m_real, m_imag = mask[:, 0], mask[:, 1]
        est_real = ref.real * m_real - ref.imag * m_imag
        est_imag = ref.real * m_imag + ref.imag * m_real
        est_complex = torch.complex(est_real, est_imag)
        
        return torch.istft(est_complex, self.n_fft, self.hop_length, window=self.window)

# ==========================================
# 4. LOSS FUNCTIONS 
# ==========================================

# A. Standard SI-SDR Loss
class SpatialSeparationLoss(nn.Module):
    def __init__(self, n_fft=512, hop_length=160):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
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

    def forward(self, estimate, reference):
        min_len = min(estimate.shape[-1], reference.shape[-1])
        estimate = estimate[..., :min_len]
        reference = reference[..., :min_len]
        ref_energy = torch.sum(reference ** 2, dim=-1)
        
        if (ref_energy > 1e-5).any():
            return self.si_sdr(estimate[ref_energy > 1e-5], reference[ref_energy > 1e-5])
        else:
            return self.mse(estimate, reference) * 1000.0

# B. NEW ENHANCED Multi-Resolution Loss (Power-Law Compressed)
class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[50, 120, 240], win_lengths=[240, 600, 1200], factor_sc=0.5, factor_mag=0.5):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def stft(self, x, fft_size, hop_size, win_length):
        return torch.stft(x, fft_size, hop_length=hop_size, win_length=win_length, 
                          window=torch.hann_window(win_length).to(x.device), 
                          return_complex=True)

    def forward(self, est, target):
        loss = 0.0
        min_len = min(est.shape[-1], target.shape[-1])
        est, target = est[..., :min_len], target[..., :min_len]

        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            est_stft = self.stft(est, n_fft, hop, win)
            tgt_stft = self.stft(target, n_fft, hop, win)
            
            # --- COMPRESSION STEP (Mimics Human Ear) ---
            est_mag = (torch.abs(est_stft) + 1e-7)
            tgt_mag = (torch.abs(tgt_stft) + 1e-7)
            
            # Power Law Compression (0.3 is standard for audio perceptual loss)
            est_c = est_mag ** 0.3
            tgt_c = tgt_mag ** 0.3

            # 1. Spectral Convergence (on Compressed Mag)
            sc_loss = torch.norm(tgt_c - est_c, p="fro") / (torch.norm(tgt_c, p="fro") + 1e-7)
            
            # 2. Log Magnitude Loss (Forces clean up of quiet noise)
            mag_loss = F.l1_loss(torch.log(tgt_mag), torch.log(est_mag))

            loss += (self.factor_sc * sc_loss) + (self.factor_mag * mag_loss)
            
        return loss / len(self.fft_sizes)

# ==========================================
# 5. FINE-TUNING LOOP
# ==========================================
def main():
    print(f"--- Fine-Tuning for Compressed PESQ on {DEVICE} ---")
    
    # 1. Dataset
    try:
        full_dataset = RoomAcousticDataset(DATASET_ROOT, silence_prob=SILENCE_PROB)
    except Exception as e:
        print(f"Error: {e}")
        return

    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # 2. Model
    model = SpatialMediumCRN(n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEVICE)
    
    if os.path.exists(PRETRAINED_PATH):
        print(f"[INFO] Loading pre-trained weights from: {PRETRAINED_PATH}")
        try:
            state_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"[ERROR] {e}")
            return
    else:
        print(f"[WARNING] Pre-trained file not found! Starting scratch.")

    # 3. Optimizer & Mixed Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    criterion_sisdr = SpatialSeparationLoss().to(DEVICE)
    criterion_mrstft = MultiResolutionSTFTLoss().to(DEVICE)
    
    best_val_loss = float('inf')
    patience_counter = 0 
    
    print(f"\nStarting Compressed Loss Fine-Tuning: {len(train_ds)} train, {len(val_ds)} validation.")
    
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss_total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        
        for mixture, angle, target in loop:
            mixture, angle, target = mixture.to(DEVICE), angle.to(DEVICE), target.to(DEVICE).squeeze(1)
            
            estimate = model(mixture, angle)
            
            # --- MIXED LOSS ---
            loss_s = criterion_sisdr(estimate, target)
            loss_m = criterion_mrstft(estimate, target)
            
            # Weighted Sum: SI-SDR + 10 * Compressed Spectral Loss
            loss = loss_s + (10.0 * loss_m)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss_total += loss.item()
            loop.set_postfix(sisdr=f"{loss_s.item():.2f}", mr_stft=f"{loss_m.item():.2f}")
            
        # Validation
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for mixture, angle, target in val_loader:
                mixture, angle, target = mixture.to(DEVICE), angle.to(DEVICE), target.to(DEVICE).squeeze(1)
                estimate = model(mixture, angle)
                
                l_s = criterion_sisdr(estimate, target)
                l_m = criterion_mrstft(estimate, target)
                val_loss = l_s + (10.0 * l_m)
                
                val_loss_total += val_loss.item()
        
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