import os
import json
import time  # <--- Added for timing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from torchmetrics.audio import ShortTimeObjectiveIntelligibility, PerceptualEvaluationSpeechQuality, ScaleInvariantSignalDistortionRatio

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "CRN_Model_FineTuned_CompressedPESQ.pth"        
SAMPLE_FOLDER = r"D:\test_dataset\test_sample_00380" 
OUTPUT_DIR = "test_inference_CRN"    
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CUSTOM ANGLE SETTING ---
# Set to None to use the ground truth angle from meta.json.
CUSTOM_ANGLE = 90.0

# ==========================================
# 2. MODEL DEFINITION
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
# 3. UTILITY FUNCTIONS
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
# 4. MAIN INFERENCE LOOP
# ==========================================
def run_inference():
    print(f"--- Running Inference on {DEVICE} ---")
    
    # 1. Load Model
    model = SpatialMediumCRN(n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEVICE)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        return
    model.eval()

    # --- SHOW MODEL STATS ---
    params = count_parameters(model)
    print(f"\n[Model Statistics]")
    print(f"Total Parameters: {params:,}")
    print(f"Model Size (MB):  {params * 4 / (1024**2):.2f} MB (approx float32)")

    # 2. Load Data & Determine Angle
    print(f"\n[Loading Sample] {SAMPLE_FOLDER}")
    mix_path = os.path.join(SAMPLE_FOLDER, "mixture.wav")
    target_path = os.path.join(SAMPLE_FOLDER, "target.wav")
    meta_path = os.path.join(SAMPLE_FOLDER, "meta.json")

    # Load Ground Truth Angle
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        ground_truth_angle = float(meta['target_angle'])

    # Determine which angle to use for inference
    if CUSTOM_ANGLE is not None:
        target_angle = float(CUSTOM_ANGLE)
        print(f"!! USING CUSTOM ANGLE: {target_angle}° (Ground Truth was {ground_truth_angle}°)")
    else:
        target_angle = ground_truth_angle
        print(f"Using Ground Truth Angle: {target_angle}°")

    # Load and prep audio
    mixture = load_audio(mix_path).unsqueeze(0).to(DEVICE)
    target = load_audio(target_path)
    if target.shape[0] > 1: target = target[0:1, :] 
    target = target.to(DEVICE)

    # Angle tensor
    angle_tensor = torch.tensor([target_angle], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 3. Run Inference (WITH TIMING)
    print("\n[Running Inference...]")
    
    # Warmup pass (optional, mostly for GPU)
    if DEVICE.type == 'cuda':
        with torch.no_grad():
            _ = model(mixture, angle_tensor)
            torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        estimate = model(mixture, angle_tensor)
        
        # Wait for GPU to finish if applicable
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
            
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Calculate Audio Duration
    audio_duration = mixture.shape[-1] / SAMPLE_RATE
    
    print(f"Inference Time:   {inference_time:.4f} seconds")
    print(f"Audio Duration:   {audio_duration:.2f} seconds")
    print(f"Real-Time Factor: {inference_time / audio_duration:.4f}x (Lower is faster)")
    
    # Align lengths for metrics
    min_len = min(estimate.shape[-1], target.shape[-1])
    est_trim = estimate[..., :min_len]
    tgt_trim = target[..., :min_len]

    # 4. Calculate Metrics
    print("\n[Calculating Metrics...]")
    if CUSTOM_ANGLE is not None and CUSTOM_ANGLE != ground_truth_angle:
        print("NOTE: Metrics may be low because you are steering away from the ground truth target.")

    pesq_metric = PerceptualEvaluationSpeechQuality(fs=SAMPLE_RATE, mode='wb').to(DEVICE)
    stoi_metric = ShortTimeObjectiveIntelligibility(fs=SAMPLE_RATE, extended=False).to(DEVICE)
    sisdr_metric = ScaleInvariantSignalDistortionRatio().to(DEVICE)

    score_pesq = pesq_metric(est_trim, tgt_trim)
    score_stoi = stoi_metric(est_trim, tgt_trim)
    score_sisdr = sisdr_metric(est_trim, tgt_trim)

    print(f"SI-SDR: {score_sisdr.item():.4f} dB")
    print(f"STOI:   {score_stoi.item():.4f}")
    print(f"PESQ:   {score_pesq.item():.4f}")

    # 5. Save Results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sample_name = os.path.basename(SAMPLE_FOLDER)
    
    # Modify filename if custom angle is used to avoid confusion
    if CUSTOM_ANGLE is not None:
        angle_str = f"_ang{int(CUSTOM_ANGLE)}"
    else:
        angle_str = ""

    out_mix_path = os.path.join(OUTPUT_DIR, f"{sample_name}{angle_str}_mixture.wav")
    out_tgt_path = os.path.join(OUTPUT_DIR, f"{sample_name}{angle_str}_target.wav")
    out_est_path = os.path.join(OUTPUT_DIR, f"{sample_name}{angle_str}_output.wav")

    torchaudio.save(out_mix_path, mixture.squeeze(0).cpu(), SAMPLE_RATE)
    torchaudio.save(out_tgt_path, tgt_trim.cpu(), SAMPLE_RATE)
    torchaudio.save(out_est_path, est_trim.cpu(), SAMPLE_RATE)

    print(f"\n[Files Saved]")
    print(f"Output:  {out_est_path}")

if __name__ == "__main__":
    run_inference()