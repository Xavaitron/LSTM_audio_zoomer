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
DATASET_ROOT = r"D:/final_reverb_fixed"
BATCH_SIZE = 4
LEARNING_RATE = 6e-5  # Scaled down for smaller batch
N_EPOCHS = 25
N_FFT = 512
HOP_LENGTH = 128  # Finer resolution for reverb
SILENCE_PROB = 0.0
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DATASET LOADER
# ==========================================
class RoomAcousticDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, fixed_length=3.0, silence_prob=0.3):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * fixed_length)
        self.silence_prob = silence_prob
        
        print(f"Scanning dataset at {root_dir}...")
        self.sample_folders = sorted(glob.glob(os.path.join(root_dir, "sample_*")))
        
        if len(self.sample_folders) == 0:
            raise ValueError(f"No 'sample_XXXXX' folders found in {root_dir}!")
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
                ground_truth = ground_truth[0:1, :]  # Force Mono

        return mixture, torch.tensor([input_angle], dtype=torch.float32), ground_truth


# ==========================================
# 3. COMPLEX CONVOLUTION MODULES
# ==========================================
class ComplexConv2d(nn.Module):
    """Complex-valued 2D convolution with coupled real/imag processing."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_real = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        
    def forward(self, x_real, x_imag):
        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return out_real, out_imag


class ComplexConvTranspose2d(nn.Module):
    """Complex-valued 2D transposed convolution."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, output_padding=0):
        super().__init__()
        self.conv_real = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding)
        self.conv_imag = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding)
        
    def forward(self, x_real, x_imag):
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return out_real, out_imag


class ComplexBatchNorm2d(nn.Module):
    """Batch normalization for complex tensors."""
    def __init__(self, num_features):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)
        
    def forward(self, x_real, x_imag):
        return self.bn_real(x_real), self.bn_imag(x_imag)


# ==========================================
# 4. SQUEEZE-EXCITATION ATTENTION
# ==========================================
class SqueezeExcitation(nn.Module):
    """Channel attention for feature recalibration."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        scale = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


# ==========================================
# 5. DUAL-PATH LSTM BLOCK
# ==========================================
class DualPathLSTM(nn.Module):
    """Processes both time and frequency dimensions for long-range dependencies."""
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.freq_lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                  batch_first=True, bidirectional=True)
        self.time_lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers,
                                  batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, input_size)
        
    def forward(self, x):
        # x: [B, C, F, T]
        B, C, F, T = x.shape
        
        # Frequency path: process each time frame across frequencies
        x_freq = x.permute(0, 3, 2, 1).reshape(B * T, F, C)  # [B*T, F, C]
        x_freq, _ = self.freq_lstm(x_freq)  # [B*T, F, 2*H]
        x_freq = x_freq.reshape(B, T, F, -1).permute(0, 2, 1, 3)  # [B, F, T, 2*H]
        
        # Time path: process each frequency bin across time
        x_time = x_freq.reshape(B * F, T, -1)  # [B*F, T, 2*H]
        x_time, _ = self.time_lstm(x_time)  # [B*F, T, 2*H]
        x_time = self.fc(x_time)  # [B*F, T, C]
        
        # Reshape back
        out = x_time.reshape(B, F, T, C).permute(0, 3, 1, 2)  # [B, C, F, T]
        return out + x  # Residual connection


# ==========================================
# 6. DCCRN++ MODEL
# ==========================================
class DCCRNpp(nn.Module):
    """Deep Complex CRN++ with dual-path attention for reverberant speech."""
    def __init__(self, n_fft=512, hop_length=128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))
        
        # Encoder channels: 2 -> 48 -> 96 -> 192 -> 384 (scaled up for ~8M params)
        self.enc1 = ComplexConv2d(2, 48, (3, 3), stride=(2, 1), padding=(1, 1))
        self.bn1 = ComplexBatchNorm2d(48)
        self.se1 = SqueezeExcitation(48)
        
        self.enc2 = ComplexConv2d(48, 96, (3, 3), stride=(2, 1), padding=(1, 1))
        self.bn2 = ComplexBatchNorm2d(96)
        self.se2 = SqueezeExcitation(96)
        
        self.enc3 = ComplexConv2d(96, 192, (3, 3), stride=(2, 1), padding=(1, 1))
        self.bn3 = ComplexBatchNorm2d(192)
        self.se3 = SqueezeExcitation(192)
        
        self.enc4 = ComplexConv2d(192, 384, (3, 3), stride=(2, 1), padding=(1, 1))
        self.bn4 = ComplexBatchNorm2d(384)
        
        # Angle conditioning network (deeper for better spatial encoding)
        self.angle_net = nn.Sequential(
            nn.Linear(2, 192),
            nn.ReLU(),
            nn.Linear(192, 384),
            nn.ReLU(),
            nn.Linear(384, 384)
        )
        
        # Dual-path LSTM bottleneck (larger hidden for longer reverb tails)
        self.dual_path = DualPathLSTM(384, 192, num_layers=2)
        
        # Decoder with skip connections (matched to encoder)
        self.dec4 = ComplexConvTranspose2d(768, 192, (3, 3), stride=(2, 1), padding=(1, 1))
        self.dbn4 = ComplexBatchNorm2d(192)
        
        self.dec3 = ComplexConvTranspose2d(384, 96, (3, 3), stride=(2, 1), padding=(1, 1))
        self.dbn3 = ComplexBatchNorm2d(96)
        
        self.dec2 = ComplexConvTranspose2d(192, 48, (3, 3), stride=(2, 1), padding=(1, 1))
        self.dbn2 = ComplexBatchNorm2d(48)
        
        self.dec1 = ComplexConvTranspose2d(96, 2, (3, 3), stride=(2, 1), padding=(1, 1))
        
        # Output mask refinement
        self.mask_conv = nn.Conv2d(4, 2, (1, 1))  # Combine real/imag for final mask

    def forward(self, x, angle):
        # STFT: x is [B, 2, samples] stereo input
        B = x.shape[0]
        x_flat = x.reshape(-1, x.shape[-1])  # [B*2, samples]
        stft = torch.stft(x_flat, self.n_fft, self.hop_length, 
                          window=self.window, return_complex=True)
        stft = stft.view(B, 2, stft.shape[-2], stft.shape[-1])  # [B, 2, F, T]
        
        x_real = stft.real
        x_imag = stft.imag
        
        # Encoder
        e1_r, e1_i = self.enc1(x_real, x_imag)
        e1_r, e1_i = self.bn1(F.leaky_relu(e1_r, 0.2), F.leaky_relu(e1_i, 0.2))
        e1_r = self.se1(e1_r)
        
        e2_r, e2_i = self.enc2(e1_r, e1_i)
        e2_r, e2_i = self.bn2(F.leaky_relu(e2_r, 0.2), F.leaky_relu(e2_i, 0.2))
        e2_r = self.se2(e2_r)
        
        e3_r, e3_i = self.enc3(e2_r, e2_i)
        e3_r, e3_i = self.bn3(F.leaky_relu(e3_r, 0.2), F.leaky_relu(e3_i, 0.2))
        e3_r = self.se3(e3_r)
        
        e4_r, e4_i = self.enc4(e3_r, e3_i)
        e4_r, e4_i = self.bn4(F.leaky_relu(e4_r, 0.2), F.leaky_relu(e4_i, 0.2))
        
        # Angle injection
        rad = torch.deg2rad(angle)
        angle_vec = torch.cat([torch.sin(rad), torch.cos(rad)], dim=1)
        angle_emb = self.angle_net(angle_vec).unsqueeze(-1).unsqueeze(-1)
        
        # Apply angle to bottleneck (FiLM-style)
        e4_r = e4_r + angle_emb
        e4_i = e4_i + angle_emb
        
        # Dual-path LSTM (process magnitude, apply to both)
        combined = torch.sqrt(e4_r**2 + e4_i**2 + 1e-8)
        combined = self.dual_path(combined)
        
        # Scale real/imag by processed features
        e4_r = e4_r * combined
        e4_i = e4_i * combined
        
        # Decoder with skip connections
        d4_r, d4_i = self.dec4(torch.cat([e4_r, e4_r], dim=1), torch.cat([e4_i, e4_i], dim=1))
        d4_r, d4_i = self._match_and_add(d4_r, d4_i, e3_r, e3_i)
        d4_r, d4_i = self.dbn4(F.leaky_relu(d4_r, 0.2), F.leaky_relu(d4_i, 0.2))
        
        d3_r, d3_i = self.dec3(torch.cat([d4_r, e3_r], dim=1), torch.cat([d4_i, e3_i], dim=1))
        d3_r, d3_i = self._match_and_add(d3_r, d3_i, e2_r, e2_i)
        d3_r, d3_i = self.dbn3(F.leaky_relu(d3_r, 0.2), F.leaky_relu(d3_i, 0.2))
        
        d2_r, d2_i = self.dec2(torch.cat([d3_r, e2_r], dim=1), torch.cat([d3_i, e2_i], dim=1))
        d2_r, d2_i = self._match_and_add(d2_r, d2_i, e1_r, e1_i)
        d2_r, d2_i = self.dbn2(F.leaky_relu(d2_r, 0.2), F.leaky_relu(d2_i, 0.2))
        
        d1_r, d1_i = self.dec1(torch.cat([d2_r, e1_r], dim=1), torch.cat([d2_i, e1_i], dim=1))
        
        # Match to input STFT size
        d1_r = self._match_size(d1_r, x_real)
        d1_i = self._match_size(d1_i, x_imag)
        
        # Complex ratio mask
        mask_input = torch.cat([d1_r, d1_i], dim=1)  # [B, 4, F, T]
        mask = self.mask_conv(mask_input)  # [B, 2, F, T]
        mask = torch.tanh(mask)  # Bounded mask
        
        m_real = mask[:, 0:1]
        m_imag = mask[:, 1:2]
        
        # Apply complex mask to reference channel (left)
        ref_real = stft[:, 0:1].real
        ref_imag = stft[:, 0:1].imag
        
        est_real = ref_real * m_real - ref_imag * m_imag
        est_imag = ref_real * m_imag + ref_imag * m_real
        
        est_stft = torch.complex(est_real.squeeze(1), est_imag.squeeze(1))
        
        # ISTFT
        output = torch.istft(est_stft, self.n_fft, self.hop_length, window=self.window)
        return output
    
    def _match_size(self, x, target):
        """Match spatial dimensions of x to target."""
        if x.shape[-2:] != target.shape[-2:]:
            x = F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        return x
    
    def _match_and_add(self, x_r, x_i, skip_r, skip_i):
        """Match sizes and add skip connections."""
        x_r = self._match_size(x_r, skip_r)
        x_i = self._match_size(x_i, skip_i)
        return x_r, x_i


# ==========================================
# 7. SI-SDR + PERCEPTUAL LOSS
# ==========================================
class SISdrPerceptualLoss(nn.Module):
    """Loss function with SI-SDR + Mel perceptual loss for PESQ improvement."""
    def __init__(self, n_fft=512, hop_length=128, alpha_sisdr=10.0, alpha_spectral=1.0, alpha_mel=2.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha_sisdr = alpha_sisdr
        self.alpha_spectral = alpha_spectral
        self.alpha_mel = alpha_mel
        self.register_buffer('window', torch.hann_window(n_fft))
        self.mse = nn.MSELoss()
        
        # Mel spectrogram for perceptual loss
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80
        )

    def si_sdr(self, estimate, reference):
        eps = 1e-8
        est = estimate - torch.mean(estimate, dim=-1, keepdim=True)
        ref = reference - torch.mean(reference, dim=-1, keepdim=True)
        ref_energy = torch.sum(ref ** 2, dim=-1, keepdim=True) + eps
        projection = torch.sum(est * ref, dim=-1, keepdim=True) * ref / ref_energy
        noise = est - projection
        ratio = torch.sum(projection ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
        return -10 * torch.log10(ratio + eps).mean()

    def multi_resolution_stft_loss(self, estimate, reference):
        """Multi-resolution STFT loss for better frequency coverage."""
        total_loss = 0
        for n_fft in [512, 1024, 2048]:
            hop = n_fft // 4
            window = torch.hann_window(n_fft, device=estimate.device)
            
            est_stft = torch.stft(estimate, n_fft, hop, window=window, return_complex=True)
            ref_stft = torch.stft(reference, n_fft, hop, window=window, return_complex=True)
            
            # Magnitude loss
            mag_loss = F.l1_loss(torch.abs(est_stft), torch.abs(ref_stft))
            
            # Log magnitude loss (emphasizes quiet parts)
            log_mag_loss = F.l1_loss(
                torch.log(torch.abs(est_stft) + 1e-8),
                torch.log(torch.abs(ref_stft) + 1e-8)
            )
            
            total_loss += mag_loss + log_mag_loss
        return total_loss / 3

    def mel_perceptual_loss(self, estimate, reference):
        """Mel-spectrogram loss as differentiable PESQ proxy."""
        # Move mel transform to correct device
        mel_transform = self.mel_transform.to(estimate.device)
        
        est_mel = torch.log(mel_transform(estimate) + 1e-8)
        ref_mel = torch.log(mel_transform(reference) + 1e-8)
        return F.l1_loss(est_mel, ref_mel)

    def forward(self, estimate, reference):
        min_len = min(estimate.shape[-1], reference.shape[-1])
        estimate = estimate[..., :min_len]
        reference = reference[..., :min_len]

        ref_energy = torch.sum(reference ** 2, dim=-1)
        has_speech = ref_energy > 1e-5
        
        total_loss = torch.tensor(0.0, device=estimate.device)
        count = 0

        if has_speech.any():
            l_sisdr = self.si_sdr(estimate[has_speech], reference[has_speech])
            l_stft = self.multi_resolution_stft_loss(estimate[has_speech], reference[has_speech])
            l_mel = self.mel_perceptual_loss(estimate[has_speech], reference[has_speech])
            total_loss += (self.alpha_sisdr * l_sisdr) + (self.alpha_spectral * l_stft) + (self.alpha_mel * l_mel)
            count += 1
            
        if (~has_speech).any():
            l_silence = self.mse(estimate[~has_speech], reference[~has_speech]) * 500.0
            total_loss += l_silence
            count += 1

        return total_loss / max(count, 1)


# ==========================================
# 8. TRAINING LOOP
# ==========================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print(f"--- DCCRN++ Spatial Audio Training on {DEVICE} ---")
    
    full_dataset = RoomAcousticDataset(DATASET_ROOT, silence_prob=SILENCE_PROB)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS)
    
    # Initialize DCCRN++ model
    model = DCCRNpp(n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEVICE)
    
    # Count and verify parameters
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    assert num_params < 10_000_000, f"Model exceeds 10M params: {num_params:,}"
    
    # AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)
    
    # SI-SDR + Perceptual loss (with Mel for PESQ improvement)
    criterion = SISdrPerceptualLoss(n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEVICE)
    
    best_val_loss = float('inf')
    
    print(f"Starting Training: {len(train_ds)} train, {len(val_ds)} validation.")
    
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss_total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        
        for mixture, angle, target in loop:
            mixture = mixture.to(DEVICE)
            angle = angle.to(DEVICE)
            target = target.to(DEVICE).squeeze(1)
            
            estimate = model(mixture, angle)
            loss = criterion(estimate, target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss_total += loss.item()
            loop.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for mixture, angle, target in val_loader:
                mixture = mixture.to(DEVICE)
                angle = angle.to(DEVICE)
                target = target.to(DEVICE).squeeze(1)
                estimate = model(mixture, angle)
                val_loss_total += criterion(estimate, target).item()
        
        avg_val_loss = val_loss_total / len(val_loader)
        avg_train_loss = train_loss_total / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "DCCRN_Reverb.pth")
            print(">>> New Best Model Saved!")
            
    print("Training Complete.")


if __name__ == "__main__":
    main()
