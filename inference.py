import os
import shutil
import time
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pystoi import stoi
from pesq import pesq
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "LSTM_model.pth"

# Input Folder (Change this to test different samples)
TEST_FOLDER = r"C:\Users\ironp\Downloads\SP Cup\LSTM_model\anechoic_dataset\sample_123456"

# Output Configuration
OUTPUT_ROOT = "inference_output"  # Main folder for results

# Model Config (Must match training)
N_FFT = 512
HOP_LENGTH = 160
HIDDEN_SIZE = 320
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths derived from TEST_FOLDER
MIXTURE_PATH = os.path.join(TEST_FOLDER, "mixture.wav")
TARGET_PATH = os.path.join(TEST_FOLDER, "target.wav")
SAMPLE_NAME = os.path.basename(TEST_FOLDER) 

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class StatefulStereoMaskNet(nn.Module):
    def __init__(self, n_fft=512, hidden_size=320, num_layers=2):
        super(StatefulStereoMaskNet, self).__init__()
        self.n_fft = n_fft
        self.freq_bins = n_fft // 2 + 1
        input_dim = self.freq_bins * 3 
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, self.freq_bins)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_left_stft, x_right_stft):
        mag_left = torch.abs(x_left_stft)
        log_mag = torch.log1p(mag_left)
        
        phase_left = torch.angle(x_left_stft)
        phase_right = torch.angle(x_right_stft)
        ipd = phase_left - phase_right
        cos_ipd, sin_ipd = torch.cos(ipd), torch.sin(ipd)
        
        x = torch.cat([log_mag, cos_ipd, sin_ipd], dim=1).permute(0, 2, 1) 
        lstm_out, _ = self.lstm(x)
        mask = self.sigmoid(self.fc(lstm_out)).permute(0, 2, 1)
        return x_left_stft * mask, mask

# ==========================================
# 3. BENCHMARK UTILS
# ==========================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_metrics(clean_wav, est_wav, sr=16000):
    clean_np = clean_wav.squeeze().cpu().numpy()
    est_np = est_wav.squeeze().cpu().numpy()
    
    # SI-SDR
    si_sdr_func = ScaleInvariantSignalDistortionRatio().to(clean_wav.device)
    sisdr_score = si_sdr_func(est_wav, clean_wav).item()
    
    # STOI
    stoi_score = stoi(clean_np, est_np, sr, extended=False)
    
    # PESQ
    try:
        pesq_score = pesq(sr, clean_np, est_np, 'wb')
    except:
        pesq_score = 0.0
        
    return sisdr_score, stoi_score, pesq_score

# ==========================================
# 4. MAIN RUN
# ==========================================
def main():
    print(f"--- Benchmarking on {DEVICE} ---")
    
    # A. Setup Output Directory
    save_dir = os.path.join(OUTPUT_ROOT, SAMPLE_NAME)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created output directory: {save_dir}")
    
    # B. Load Model
    model = StatefulStereoMaskNet(n_fft=N_FFT, hidden_size=HIDDEN_SIZE).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("Model file not found!")
        return
    model.eval()

    # C. Load Audio
    if not os.path.exists(MIXTURE_PATH) or not os.path.exists(TARGET_PATH):
        print("Audio files not found. Check paths.")
        return

    mix_wav, sr = torchaudio.load(MIXTURE_PATH)
    target_wav, _ = torchaudio.load(TARGET_PATH)
    
    mix_wav = mix_wav.to(DEVICE)
    target_wav = target_wav.to(DEVICE)
    window = torch.hann_window(N_FFT).to(DEVICE)

    # D. Measure Latency
    inputs = mix_wav.unsqueeze(0)
    stft_l = torch.stft(inputs[:, 0], N_FFT, HOP_LENGTH, window=window, return_complex=True)
    stft_r = torch.stft(inputs[:, 1], N_FFT, HOP_LENGTH, window=window, return_complex=True)
    
    # Warmup
    for _ in range(5): 
        with torch.no_grad(): _ = model(stft_l, stft_r)
    
    # Timing
    start_time = time.time()
    for _ in range(50):
        with torch.no_grad(): _ = model(stft_l, stft_r)
    avg_time_ms = ((time.time() - start_time) / 50) * 1000
    rtf = avg_time_ms / ((mix_wav.shape[1] / sr) * 1000)

    # E. Run Inference
    with torch.no_grad():
        sep_stft, _ = model(stft_l, stft_r)
        sep_wav = torch.istft(sep_stft, N_FFT, HOP_LENGTH, window=window, return_complex=False)
        sep_wav = sep_wav.cpu()
        target_wav = target_wav.cpu()
    
    # Match Lengths
    min_len = min(sep_wav.shape[-1], target_wav.shape[-1])
    sep_wav = sep_wav[..., :min_len]
    target_wav = target_wav[..., :min_len]

    # F. Metrics
    sisdr, stoi_score, pesq_score = calculate_metrics(target_wav, sep_wav, sr)

    # ==========================================
    # 5. SAVING EVERYTHING
    # ==========================================
    print(f"\n--- Saving Results to: {save_dir} ---")
    
    # 1. Save Enhanced Audio
    out_wav_path = os.path.join(save_dir, "enhanced.wav")
    torchaudio.save(out_wav_path, sep_wav, sr)
    print(f"[Saved] Enhanced Audio")

    # 2. Copy Original Files
    shutil.copy(MIXTURE_PATH, os.path.join(save_dir, "mixture.wav"))
    shutil.copy(TARGET_PATH, os.path.join(save_dir, "target.wav"))
    print(f"[Copied] Mixture & Target")

    # 3. Save Stats Text File
    stats_path = os.path.join(save_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Sample: {SAMPLE_NAME}\n")
        f.write(f"Model Parameters: {count_parameters(model):,}\n")
        f.write("-" * 20 + "\n")
        f.write(f"SI-SDR: {sisdr:.2f} dB\n")
        f.write(f"PESQ:   {pesq_score:.2f}\n")
        f.write(f"STOI:   {stoi_score:.2f}\n")
        f.write("-" * 20 + "\n")
        f.write(f"Inference Time: {avg_time_ms:.2f} ms\n")
        f.write(f"RTF: {rtf:.4f}\n")
    
    print(f"[Saved] Stats.txt")
    
    # Print to Console
    print(f"\nFINAL SCORES:")
    print(f"SI-SDR: {sisdr:.2f} dB")
    print(f"PESQ:   {pesq_score:.2f}")
    print(f"STOI:   {stoi_score:.2f}")

if __name__ == "__main__":
    main()