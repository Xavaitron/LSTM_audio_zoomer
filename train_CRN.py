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
# 1. CONFIGURATION
# ==========================================
# Path to your Dataset (Folder containing sample_00001, etc.)
DATASET_ROOT = r"D:\your_dataset_folder" 

# Hyperparameters
BATCH_SIZE = 32          # Adjusted for mobile model size
LEARNING_RATE = 1e-3     # Standard Adam LR
N_EPOCHS = 60            # Max number of epochs
PATIENCE = 10            # Early Stopping Patience
N_FFT = 512              # Number of FFT bins
HOP_LENGTH = 160         # 10ms at 16kHz
HIDDEN_SIZE = 128        # GRU Hidden Size
NUM_WORKERS = 4          # CPU threads for data loading
SILENCE_PROB = 0.3       # Probability of training on empty angles (Negative Sampling)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. DATASET LOADER (With Angle & Silence)
# ==========================================
class RoomAcousticDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, fixed_length=3.0, silence_prob=0.3):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * fixed_length)
        self.silence_prob = silence_prob
        
        # Fast scanning of sample directories
        print(f"Scanning dataset at {root_dir}...")
        self.sample_folders = sorted(glob.glob(os.path.join(root_dir, "sample_*")))
        
        if len(self.sample_folders) == 0:
            raise ValueError(f"No 'sample_XXXXX' folders found in {root_dir}!")
            
        print(f"Found {len(self.sample_folders)} samples. Silence Prob: {silence_prob}")

    def __len__(self):
        return len(self.sample_folders)

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        
        # Resample
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Pad or Crop to fixed length
        if waveform.shape[-1] < self.num_samples:
            pad_amt = self.num_samples - waveform.shape[-1]
            waveform = F.pad(waveform, (0, pad_amt))
        else:
            waveform = waveform[:, :self.num_samples]
            
        return waveform

    def __getitem__(self, idx):
        folder_path = self.folders[idx] if hasattr(self, 'folders') else self.sample_folders[idx]
        
        mix_path = os.path.join(folder_path, "mixture.wav")
        target_path = os.path.join(folder_path, "target.wav")
        meta_path = os.path.join(folder_path, "meta.json")
        
        # Load Metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            target_angle = float(meta['target_angle'])
            interf_angle = float(meta['interf_angle']) # Needed to avoid picking interference angle during silence training
            
        # Load Mixture
        mixture = self._load_audio(mix_path)
        
        # --- NEGATIVE SAMPLING LOGIC ---
        # 30% of the time, pick a random angle far from sources and expect Silence
        if random.random() < self.silence_prob:
            valid_angle = False
            while not valid_angle:
                random_angle = random.uniform(0, 180)
                # Ensure we don't accidentally point at the target or interference
                if abs(random_angle - target_angle) > 20 and abs(random_angle - interf_angle) > 20:
                    input_angle = random_angle
                    valid_angle = True
            
            # Ground Truth is Silence
            ground_truth = torch.zeros(1, self.num_samples)
        else:
            # Normal Case: Point at Target
            input_angle = target_angle
            ground_truth = self._load_audio(target_path)
            if ground_truth.shape[0] > 1:
                ground_truth = ground_truth[0:1, :] # Force Mono Target

        return mixture, torch.tensor([input_angle], dtype=torch.float32), ground_truth


# ==========================================
# 3. SPATIAL TINY-CRN MODEL
# ==========================================
class SpatialTinyCRN(nn.Module):
    def __init__(self, n_fft=512, hop_length=160):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))
        
        # Encoder (2ch audio -> 4ch features -> deep features)
        # Input: [Batch, 4, Freq, Time] (4 = Real_L, Imag_L, Real_R, Imag_R)
        self.enc1 = nn.Conv2d(4, 16, (3,3), stride=(2,1), padding=(1,1))
        self.enc2 = nn.Conv2d(16, 32, (3,3), stride=(2,1), padding=(1,1))
        self.enc3 = nn.Conv2d(32, 64, (3,3), stride=(2,1), padding=(1,1))
        self.enc4 = nn.Conv2d(64, 128, (3,3), stride=(2,1), padding=(1,1))
        
        # Angle Injection Network
        self.angle_net = nn.Sequential(
            nn.Linear(2, 64), 
            nn.ReLU(), 
            nn.Linear(64, 128)
        )
        
        # Bottleneck GRU
        # With n_fft=512, freq bins=257. After 4 layers of stride 2: 257->129->65->33->17
        self.gru = nn.GRU(128 * 17, 128, batch_first=True)
        self.gru_fc = nn.Linear(128, 128 * 17)
        
        # Decoder
        self.dec4 = nn.ConvTranspose2d(256, 64, (3,3), stride=(2,1), padding=(1,1), output_padding=(1,0))
        self.dec3 = nn.ConvTranspose2d(128, 32, (3,3), stride=(2,1), padding=(1,1), output_padding=(1,0))
        self.dec2 = nn.ConvTranspose2d(64, 16, (3,3), stride=(2,1), padding=(1,1), output_padding=(1,0))
        self.dec1 = nn.ConvTranspose2d(32, 2, (3,3), stride=(2,1), padding=(1,1), output_padding=(1,0))

    def forward(self, x, angle):
        # x: [Batch, 2, Time]
        # angle: [Batch, 1] (Degrees)
        
        # STFT
        stft = torch.stft(x.reshape(-1, x.shape[-1]), self.n_fft, self.hop_length, window=self.window, return_complex=True)
        stft = stft.view(x.shape[0], 2, stft.shape[-2], stft.shape[-1])
        
        # Prepare Input: Concat Real/Imag of both mics [B, 4, F, T]
        feat = torch.cat([stft.real, stft.imag], dim=1)
        
        # Encoding
        e1 = F.elu(self.enc1(feat))
        e2 = F.elu(self.enc2(e1))
        e3 = F.elu(self.enc3(e2))
        e4 = F.elu(self.enc4(e3)) # [B, 128, 17, T]
        
        # Angle Embedding
        rad = torch.deg2rad(angle)
        angle_vec = torch.cat([torch.sin(rad), torch.cos(rad)], dim=1) # [B, 2]
        angle_emb = self.angle_net(angle_vec).unsqueeze(-1).unsqueeze(-1) # [B, 128, 1, 1]
        
        # GRU Processing
        b, c, f, t = e4.shape
        gru_in = e4.permute(0, 3, 1, 2).reshape(b, t, -1)
        gru_out, _ = self.gru(gru_in)
        gru_out = F.relu(self.gru_fc(gru_out)).reshape(b, t, c, f).permute(0, 2, 3, 1)
        
        # Inject Angle Info (This acts as the "Gate")
        gru_out = gru_out + angle_emb
        
        # Decoding
        d4 = F.elu(self.dec4(torch.cat([gru_out, e4], dim=1)))
        d3 = F.elu(self.dec3(torch.cat([d4, e3], dim=1)))
        d2 = F.elu(self.dec2(torch.cat([d3, e2], dim=1)))
        mask = self.dec1(torch.cat([d2, e1], dim=1))
        
        # Apply Mask (Complex Mult)
        ref = stft[:, 0] # Use Mic 1 as reference
        m_real, m_imag = mask[:, 0], mask[:, 1]
        est_real = ref.real * m_real - ref.imag * m_imag
        est_imag = ref.real * m_imag + ref.imag * m_real
        est_complex = torch.complex(est_real, est_imag)
        
        # iSTFT
        return torch.istft(est_complex, self.n_fft, self.hop_length, window=self.window)


# ==========================================
# 4. LOSS FUNCTION (SI-SDR + Spectral + Silence MSE)
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
        
        # Normalize per time-step (Spectral Shape)
        mu_est = torch.mean(mag_est, dim=1, keepdim=True)
        std_est = torch.std(mag_est, dim=1, keepdim=True) + 1e-8
        norm_est = (mag_est - mu_est) / std_est
        
        mu_ref = torch.mean(mag_ref, dim=1, keepdim=True)
        std_ref = torch.std(mag_ref, dim=1, keepdim=True) + 1e-8
        norm_ref = (mag_ref - mu_ref) / std_ref
        
        return -torch.mean(norm_est * norm_ref)

    def forward(self, estimate, reference):
        # Align lengths
        min_len = min(estimate.shape[-1], reference.shape[-1])
        estimate = estimate[..., :min_len]
        reference = reference[..., :min_len]

        # Check for silence in the batch (Energy < Threshold)
        ref_energy = torch.sum(reference ** 2, dim=-1)
        has_speech = ref_energy > 1e-5
        
        total_loss = torch.tensor(0.0, device=estimate.device)
        count = 0

        # 1. Loss for Speech Samples (SI-SDR + Spectral)
        if has_speech.any():
            est_speech = estimate[has_speech]
            ref_speech = reference[has_speech]
            
            l_time = self.si_sdr(est_speech, ref_speech)
            l_freq = self.spectral_correlation(est_speech, ref_speech)
            total_loss += (self.alpha_sisdr * l_time) + (self.alpha_spectral * l_freq)
            count += 1
            
        # 2. Loss for Silence Samples (MSE)
        if (~has_speech).any():
            est_silence = estimate[~has_speech]
            ref_silence = reference[~has_speech] 
            
            # Weight MSE to match SI-SDR scale
            l_silence = self.mse(est_silence, ref_silence) * 1000.0 
            total_loss += l_silence
            count += 1

        return total_loss / max(count, 1)


# ==========================================
# 5. TRAINING LOOP
# ==========================================
def main():
    print(f"--- Spatial Audio Training on {DEVICE} ---")
    
    # --- Data ---
    full_dataset = RoomAcousticDataset(DATASET_ROOT, silence_prob=SILENCE_PROB)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # --- Model ---
    model = SpatialTinyCRN(n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = SpatialSeparationLoss().to(DEVICE)
    
    best_val_loss = float('inf')
    patience_counter = 0 
    
    print(f"Starting Training: {len(train_ds)} train, {len(val_ds)} validation.")
    print(f"Early Stopping Patience: {PATIENCE} epochs")
    
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss_total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        
        for batch_idx, (mixture, angle, target) in enumerate(loop):
            mixture = mixture.to(DEVICE)     # [B, 2, T]
            angle = angle.to(DEVICE)         # [B, 1]
            target = target.to(DEVICE).squeeze(1) # [B, T]
            
            # Forward
            estimate = model(mixture, angle)
            
            # Loss
            loss = criterion(estimate, target)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss_total += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # --- Validation ---
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for mixture, angle, target in val_loader:
                mixture = mixture.to(DEVICE)
                angle = angle.to(DEVICE)
                target = target.to(DEVICE).squeeze(1)
                
                estimate = model(mixture, angle)
                loss = criterion(estimate, target)
                val_loss_total += loss.item()
        
        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # --- EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "SpatialTinyCRN_Best.pth")
            print(">>> New Best Model Saved!")
        else:
            patience_counter += 1
            print(f"No improvement. Counter: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print("!!! Early stopping triggered !!!")
                break
            
    print("Training Complete.")

if __name__ == "__main__":
    main()