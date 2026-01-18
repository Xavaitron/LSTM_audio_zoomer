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
import math

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_ROOT = r"./final_reverb_fixed"
BATCH_SIZE = 4
LEARNING_RATE = 2e-5  # Lower LR for fine-tuning
N_EPOCHS = 10         # Quick fine-tune
N_FFT = 512
HOP_LENGTH = 128
SILENCE_PROB = 0.0
NUM_WORKERS = 4
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Resume training from checkpoint (set to None to train from scratch)
RESUME_FROM = "DCCRN_Conformer.pth"  # or None

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
                ground_truth = ground_truth[0:1, :]

        return mixture, torch.tensor([input_angle], dtype=torch.float32), ground_truth


# ==========================================
# 3. COMPLEX CONVOLUTION MODULES
# ==========================================
class ComplexConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_real = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        
    def forward(self, x_real, x_imag):
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return out_real, out_imag


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, output_padding=0):
        super().__init__()
        self.conv_real = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding)
        self.conv_imag = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding)
        
    def forward(self, x_real, x_imag):
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return out_real, out_imag


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)
        
    def forward(self, x_real, x_imag):
        return self.bn_real(x_real), self.bn_imag(x_imag)


class SqueezeExcitation(nn.Module):
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
# 4. CONFORMER BLOCK (Conv + Attention)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ConvModule(nn.Module):
    """Conformer convolution module with depthwise separable conv."""
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Linear(d_model, 2 * d_model)
        self.glu = nn.GLU(dim=-1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, 
                                    padding=kernel_size//2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [B, T, C]
        residual = x
        x = self.layer_norm(x)
        x = self.pointwise1(x)
        x = self.glu(x)
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.pointwise2(x)
        x = self.dropout(x)
        return residual + x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with relative positional encoding."""
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        residual = x
        x = self.layer_norm(x)
        
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return residual + out


class FeedForward(nn.Module):
    """Feed-forward module with expansion."""
    def __init__(self, d_model, expansion=2, dropout=0.1):  # 2x expansion for efficiency
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion, d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x  # Half residual for stability


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN -> MHSA -> Conv -> FFN"""
    def __init__(self, d_model, num_heads=8, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, dropout=dropout)
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForward(d_model, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.ff1(x)
        x = self.mhsa(x)
        x = self.conv(x)
        x = self.ff2(x)
        x = self.layer_norm(x)
        return x


class DualPathConformer(nn.Module):
    """Dual-path processing with Conformer blocks."""
    def __init__(self, input_size, num_blocks=3, num_heads=4, dropout=0.1):  # 3 blocks for ~10M
        super().__init__()
        self.input_size = input_size
        
        # Frequency-path Conformer
        self.freq_conformer = nn.ModuleList([
            ConformerBlock(input_size, num_heads, conv_kernel=15, dropout=dropout)
            for _ in range(num_blocks)
        ])
        
        # Time-path Conformer  
        self.time_conformer = nn.ModuleList([
            ConformerBlock(input_size, num_heads, conv_kernel=31, dropout=dropout)
            for _ in range(num_blocks)
        ])
        
        self.pos_enc = PositionalEncoding(input_size)
        
    def forward(self, x):
        # x: [B, C, F, T]
        B, C, F, T = x.shape
        
        # Frequency path
        x_freq = x.permute(0, 3, 2, 1).reshape(B * T, F, C)  # [B*T, F, C]
        x_freq = self.pos_enc(x_freq)
        for block in self.freq_conformer:
            x_freq = block(x_freq)
        x_freq = x_freq.reshape(B, T, F, C).permute(0, 2, 1, 3)  # [B, F, T, C]
        
        # Time path
        x_time = x_freq.reshape(B * F, T, C)  # [B*F, T, C]
        x_time = self.pos_enc(x_time)
        for block in self.time_conformer:
            x_time = block(x_time)
        
        out = x_time.reshape(B, F, T, C).permute(0, 3, 1, 2)  # [B, C, F, T]
        return out + x  # Residual


# ==========================================
# 5. DCCRN-CONFORMER MODEL
# ==========================================
class DCCRNConformer(nn.Module):
    """DCCRN with Conformer bottleneck for enhanced attention."""
    def __init__(self, n_fft=512, hop_length=128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))
        
        # Encoder: 2 -> 48 -> 96 -> 192 -> 256 (balanced for ~10M)
        self.enc1 = ComplexConv2d(2, 48, (3, 3), stride=(2, 1), padding=(1, 1))
        self.bn1 = ComplexBatchNorm2d(48)
        self.se1 = SqueezeExcitation(48)
        
        self.enc2 = ComplexConv2d(48, 96, (3, 3), stride=(2, 1), padding=(1, 1))
        self.bn2 = ComplexBatchNorm2d(96)
        self.se2 = SqueezeExcitation(96)
        
        self.enc3 = ComplexConv2d(96, 192, (3, 3), stride=(2, 1), padding=(1, 1))
        self.bn3 = ComplexBatchNorm2d(192)
        self.se3 = SqueezeExcitation(192)
        
        self.enc4 = ComplexConv2d(192, 256, (3, 3), stride=(2, 1), padding=(1, 1))
        self.bn4 = ComplexBatchNorm2d(256)
        
        # Angle conditioning
        self.angle_net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Dual-path Conformer bottleneck
        self.conformer = DualPathConformer(256, num_blocks=3, num_heads=4, dropout=0.1)
        
        # Decoder (matched to encoder)
        self.dec4 = ComplexConvTranspose2d(512, 192, (3, 3), stride=(2, 1), padding=(1, 1))
        self.dbn4 = ComplexBatchNorm2d(192)
        
        self.dec3 = ComplexConvTranspose2d(384, 96, (3, 3), stride=(2, 1), padding=(1, 1))
        self.dbn3 = ComplexBatchNorm2d(96)
        
        self.dec2 = ComplexConvTranspose2d(192, 48, (3, 3), stride=(2, 1), padding=(1, 1))
        self.dbn2 = ComplexBatchNorm2d(48)
        
        self.dec1 = ComplexConvTranspose2d(96, 2, (3, 3), stride=(2, 1), padding=(1, 1))
        
        self.mask_conv = nn.Conv2d(4, 2, (1, 1))

    def forward(self, x, angle):
        B = x.shape[0]
        x_flat = x.reshape(-1, x.shape[-1])
        stft = torch.stft(x_flat, self.n_fft, self.hop_length, 
                          window=self.window, return_complex=True)
        stft = stft.view(B, 2, stft.shape[-2], stft.shape[-1])
        
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
        
        e4_r = e4_r + angle_emb
        e4_i = e4_i + angle_emb
        
        # Conformer bottleneck (process magnitude)
        combined = torch.sqrt(e4_r**2 + e4_i**2 + 1e-8)
        combined = self.conformer(combined)
        
        e4_r = e4_r * combined
        e4_i = e4_i * combined
        
        # Decoder
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
        
        d1_r = self._match_size(d1_r, x_real)
        d1_i = self._match_size(d1_i, x_imag)
        
        mask_input = torch.cat([d1_r, d1_i], dim=1)
        mask = self.mask_conv(mask_input)
        mask = torch.tanh(mask)
        
        m_real = mask[:, 0:1]
        m_imag = mask[:, 1:2]
        
        ref_real = stft[:, 0:1].real
        ref_imag = stft[:, 0:1].imag
        
        est_real = ref_real * m_real - ref_imag * m_imag
        est_imag = ref_real * m_imag + ref_imag * m_real
        
        est_stft = torch.complex(est_real.squeeze(1), est_imag.squeeze(1))
        
        return torch.istft(est_stft, self.n_fft, self.hop_length, window=self.window)
    
    def _match_size(self, x, target):
        if x.shape[-2:] != target.shape[-2:]:
            x = F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        return x
    
    def _match_and_add(self, x_r, x_i, skip_r, skip_i):
        x_r = self._match_size(x_r, skip_r)
        x_i = self._match_size(x_i, skip_i)
        return x_r, x_i


# ==========================================
# 6. SI-SDR + PERCEPTUAL + PHASE LOSS (AGGRESSIVE PESQ)
# ==========================================
class SISdrPerceptualLoss(nn.Module):
    def __init__(self, n_fft=512, hop_length=128, alpha_sisdr=10.0, alpha_spectral=1.0, alpha_mel=8.0, alpha_phase=1.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha_sisdr = alpha_sisdr
        self.alpha_spectral = alpha_spectral
        self.alpha_mel = alpha_mel
        self.alpha_phase = alpha_phase
        self.register_buffer('window', torch.hann_window(n_fft))
        self.mse = nn.MSELoss()
        # Increased mel bands for finer perceptual resolution
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=256, n_mels=128
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
        total_loss = 0
        for n_fft in [512, 1024, 2048]:
            hop = n_fft // 4
            window = torch.hann_window(n_fft, device=estimate.device)
            est_stft = torch.stft(estimate, n_fft, hop, window=window, return_complex=True)
            ref_stft = torch.stft(reference, n_fft, hop, window=window, return_complex=True)
            mag_loss = F.l1_loss(torch.abs(est_stft), torch.abs(ref_stft))
            log_mag_loss = F.l1_loss(
                torch.log(torch.abs(est_stft) + 1e-8),
                torch.log(torch.abs(ref_stft) + 1e-8)
            )
            total_loss += mag_loss + log_mag_loss
        return total_loss / 3

    def mel_perceptual_loss(self, estimate, reference):
        mel_transform = self.mel_transform.to(estimate.device)
        est_mel = torch.log(mel_transform(estimate) + 1e-8)
        ref_mel = torch.log(mel_transform(reference) + 1e-8)
        return F.l1_loss(est_mel, ref_mel)

    def phase_loss(self, estimate, reference):
        """Phase-aware loss - critical for PESQ."""
        total_loss = 0
        for n_fft in [512, 1024]:
            hop = n_fft // 4
            window = torch.hann_window(n_fft, device=estimate.device)
            est_stft = torch.stft(estimate, n_fft, hop, window=window, return_complex=True)
            ref_stft = torch.stft(reference, n_fft, hop, window=window, return_complex=True)
            
            # Instantaneous phase difference loss
            est_phase = torch.angle(est_stft)
            ref_phase = torch.angle(ref_stft)
            
            # Use cosine of phase difference (handles wraparound)
            phase_diff = torch.cos(est_phase - ref_phase)
            total_loss += (1 - phase_diff.mean())  # Minimize when phases align
            
        return total_loss / 2

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
            l_phase = self.phase_loss(estimate[has_speech], reference[has_speech])
            total_loss += (self.alpha_sisdr * l_sisdr) + (self.alpha_spectral * l_stft) + (self.alpha_mel * l_mel) + (self.alpha_phase * l_phase)
            count += 1
            
        if (~has_speech).any():
            l_silence = self.mse(estimate[~has_speech], reference[~has_speech]) * 500.0
            total_loss += l_silence
            count += 1

        return total_loss / max(count, 1)


# ==========================================
# 7. TRAINING LOOP
# ==========================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print(f"--- DCCRN-Conformer Training on {DEVICE} ---")
    
    full_dataset = RoomAcousticDataset(DATASET_ROOT, silence_prob=SILENCE_PROB)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS)
    
    # Initialize DCCRN-Conformer model
    model = DCCRNConformer(n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEVICE)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    assert num_params < 15_000_000, f"Model exceeds 15M params: {num_params:,}"
    
    # Load checkpoint if resuming
    if RESUME_FROM is not None:
        try:
            state_dict = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"Resumed from checkpoint: {RESUME_FROM}")
        except FileNotFoundError:
            print(f"Checkpoint not found: {RESUME_FROM}, training from scratch.")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)
    criterion = SISdrPerceptualLoss(n_fft=N_FFT, hop_length=HOP_LENGTH, alpha_mel=5.0).to(DEVICE)  # Bumped Mel for PESQ
    
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
            torch.save(model.state_dict(), "DCCRN_Conformer.pth")
            print(">>> New Best Model Saved!")
            
    print("Training Complete.")


if __name__ == "__main__":
    main()
