import os
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from torchmetrics.audio import ShortTimeObjectiveIntelligibility, PerceptualEvaluationSpeechQuality, ScaleInvariantSignalDistortionRatio
from tqdm import tqdm
import time
import math

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "DCCRN_Conformer.pth"       
TEST_DATASET_ROOT = r"D:\test_dataset"
OUTPUT_DIR = "evaluation_Conformer"    
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
DEVICE = torch.device("cpu")

# ==========================================
# 2. MODEL COMPONENTS
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
# 3. CONFORMER COMPONENTS
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
        residual = x
        x = self.layer_norm(x)
        x = self.pointwise1(x)
        x = self.glu(x)
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        x = self.pointwise2(x)
        x = self.dropout(x)
        return residual + x


class MultiHeadSelfAttention(nn.Module):
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
        B, T, C = x.shape
        residual = x
        x = self.layer_norm(x)
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        return residual + out


class FeedForward(nn.Module):
    def __init__(self, d_model, expansion=2, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion)
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
        return residual + 0.5 * x


class ConformerBlock(nn.Module):
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
    def __init__(self, input_size, num_blocks=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.freq_conformer = nn.ModuleList([
            ConformerBlock(input_size, num_heads, conv_kernel=15, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.time_conformer = nn.ModuleList([
            ConformerBlock(input_size, num_heads, conv_kernel=31, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.pos_enc = PositionalEncoding(input_size)
        
    def forward(self, x):
        B, C, F, T = x.shape
        x_freq = x.permute(0, 3, 2, 1).reshape(B * T, F, C)
        x_freq = self.pos_enc(x_freq)
        for block in self.freq_conformer:
            x_freq = block(x_freq)
        x_freq = x_freq.reshape(B, T, F, C).permute(0, 2, 1, 3)
        x_time = x_freq.reshape(B * F, T, C)
        x_time = self.pos_enc(x_time)
        for block in self.time_conformer:
            x_time = block(x_time)
        out = x_time.reshape(B, F, T, C).permute(0, 3, 1, 2)
        return out + x


# ==========================================
# 4. DCCRN-CONFORMER MODEL
# ==========================================
class DCCRNConformer(nn.Module):
    def __init__(self, n_fft=512, hop_length=128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))
        
        # Encoder: 2 -> 48 -> 96 -> 192 -> 256
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
        
        # Decoder
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
        
        # Conformer bottleneck
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
# 5. UTILITY FUNCTIONS
# ==========================================
def load_audio(path, target_len=None):
    waveform, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    
    if target_len:
        if waveform.shape[-1] < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]))
        else:
            waveform = waveform[:, :target_len]
            
    return waveform


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================
# 6. MAIN EVALUATION LOOP
# ==========================================
def run_evaluation():
    print(f"--- Running DCCRN-Conformer Evaluation on {DEVICE} ---")
    
    # 1. Load Model
    model = DCCRNConformer(n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEVICE)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Model loaded: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        return
    model.eval()
    
    # Print model info
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # 2. Setup Metrics
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=SAMPLE_RATE, mode='wb').to(DEVICE)
    stoi_metric = ShortTimeObjectiveIntelligibility(fs=SAMPLE_RATE, extended=False).to(DEVICE)
    sisdr_metric = ScaleInvariantSignalDistortionRatio().to(DEVICE)

    # 3. Find Samples
    sample_folders = sorted(glob.glob(os.path.join(TEST_DATASET_ROOT, "*sample_*")))
    
    if len(sample_folders) == 0:
        print(f"No samples found in {TEST_DATASET_ROOT}")
        return

    print(f"Found {len(sample_folders)} samples. Starting...")
    
    # Storage for Averages
    results = {'sisdr': [], 'stoi': [], 'pesq': []}
    inference_times = []
    sample_names = []

    # 4. Loop
    for folder_path in tqdm(sample_folders):
        sample_name = os.path.basename(folder_path)
        sample_names.append(sample_name)
        mix_path = os.path.join(folder_path, "mixture.wav")
        target_path = os.path.join(folder_path, "target.wav")
        meta_path = os.path.join(folder_path, "meta.json")

        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                target_angle = float(meta['target_angle'])

            mixture = load_audio(mix_path).unsqueeze(0).to(DEVICE)
            target = load_audio(target_path)
            if target.shape[0] > 1: target = target[0:1, :] 
            target = target.to(DEVICE)

            angle_tensor = torch.tensor([target_angle], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                start_time = time.perf_counter()
                estimate = model(mixture, angle_tensor)
                end_time = time.perf_counter()
                inference_times.append(end_time - start_time)
                
                min_len = min(estimate.shape[-1], target.shape[-1])
                est_trim = estimate[..., :min_len]
                tgt_trim = target[..., :min_len]

                s_pesq = pesq_metric(est_trim, tgt_trim).item()
                s_stoi = stoi_metric(est_trim, tgt_trim).item()
                s_sisdr = sisdr_metric(est_trim, tgt_trim).item()
                
                results['pesq'].append(s_pesq)
                results['stoi'].append(s_stoi)
                results['sisdr'].append(s_sisdr)

        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
            continue

    # 5. Report Final Results
    avg_sisdr = np.mean(results['sisdr'])
    avg_stoi = np.mean(results['stoi'])
    avg_pesq = np.mean(results['pesq'])

    sisdr_arr = np.array(results['sisdr'])
    stoi_arr = np.array(results['stoi'])
    pesq_arr = np.array(results['pesq'])

    def normalize(arr):
        if arr.max() == arr.min(): return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    norm_sisdr = normalize(sisdr_arr)
    norm_stoi = normalize(stoi_arr)
    norm_pesq = normalize(pesq_arr)

    combined_score = norm_sisdr + norm_stoi + norm_pesq
    best_idx = np.argmax(combined_score)
    best_sample_name = sample_names[best_idx]
    
    best_sisdr = sisdr_arr[best_idx]
    best_stoi = stoi_arr[best_idx]
    best_pesq = pesq_arr[best_idx]

    print("\n" + "="*40)
    print("   DCCRN-CONFORMER EVALUATION REPORT")
    print("="*40)
    print(f"Total Samples:   {len(results['sisdr'])}")
    print("-" * 40)
    print(f"AVERAGE SI-SDR:  {avg_sisdr:.4f} dB")
    print(f"AVERAGE STOI:    {avg_stoi:.4f}")
    print(f"AVERAGE PESQ:    {avg_pesq:.4f}")
    print("="*40)
    print("   BEST OVERALL CASE (Combined Metric)")
    print("="*40)
    print(f"Sample:          {best_sample_name}")
    print(f"Combined Score:  {combined_score[best_idx]:.4f} / 3.0")
    print(f"SI-SDR:          {best_sisdr:.4f} dB")
    print(f"STOI:            {best_stoi:.4f}")
    print(f"PESQ:            {best_pesq:.4f}")
    print("="*40)
    print("   INFERENCE STATISTICS")
    print("="*40)
    avg_inference_time = np.mean(inference_times) * 1000
    std_inference_time = np.std(inference_times) * 1000
    print(f"Avg Inference:   {avg_inference_time:.2f} ms")
    print(f"Std Inference:   {std_inference_time:.2f} ms")
    print(f"Real-time Factor: {(3.0 * 1000) / avg_inference_time:.2f}x (for 3s audio)")
    print("-" * 40)
    
    # Save Best Case
    print(f"Saving Best Overall Case: {best_sample_name}...")
    best_folder = sample_folders[best_idx]
    mix_path = os.path.join(best_folder, "mixture.wav")
    target_path = os.path.join(best_folder, "target.wav")
    meta_path = os.path.join(best_folder, "meta.json")
    
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            target_angle = float(meta['target_angle'])
            
        mixture = load_audio(mix_path).unsqueeze(0).to(DEVICE)
        target = load_audio(target_path)
        if target.shape[0] > 1: target = target[0:1, :] 
        target = target.to(DEVICE)
        
        angle_tensor = torch.tensor([target_angle], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            estimate = model(mixture, angle_tensor)
            min_len = min(estimate.shape[-1], target.shape[-1])
            est_trim = estimate[..., :min_len]
            tgt_trim = target[..., :min_len]
            
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            torchaudio.save(os.path.join(OUTPUT_DIR, f"BEST_OVERALL_output.wav"), est_trim.cpu(), SAMPLE_RATE)
            torchaudio.save(os.path.join(OUTPUT_DIR, f"BEST_OVERALL_mixture.wav"), mixture.squeeze(0).cpu(), SAMPLE_RATE)
            torchaudio.save(os.path.join(OUTPUT_DIR, f"BEST_OVERALL_target.wav"), tgt_trim.cpu(), SAMPLE_RATE)
            print(f"Saved audio files to {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error saving best case: {e}")

if __name__ == "__main__":
    run_evaluation()
