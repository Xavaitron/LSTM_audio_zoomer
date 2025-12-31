import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from torchmetrics.audio import ShortTimeObjectiveIntelligibility, PerceptualEvaluationSpeechQuality, ScaleInvariantSignalDistortionRatio

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "CRN_Model.pth"        # Path to your trained model checkpoint
SAMPLE_FOLDER = r"D:\anechoic_dataset_v3\sample_00072" # Change to specific sample folder
OUTPUT_DIR = "inference_CRN"    # Where to save the output files
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    # Handle length if needed 
    if target_len:
        if waveform.shape[-1] < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]))
        else:
            waveform = waveform[:, :target_len]
            
    return waveform

def get_model_size_mb(model):
    """Calculates the model size in MB based on parameters."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

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
        # Added weights_only=True to silence the FutureWarning
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        return
    model.eval()

    # 2. Calculate Model Stats
    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    print(f"\n[Model Stats]")
    print(f"Parameter Count: {params:,}")
    print(f"Model Size (approx): {size_mb:.2f} MB")

    # 3. Load Data
    print(f"\n[Loading Sample] {SAMPLE_FOLDER}")
    mix_path = os.path.join(SAMPLE_FOLDER, "mixture.wav")
    target_path = os.path.join(SAMPLE_FOLDER, "target.wav")
    meta_path = os.path.join(SAMPLE_FOLDER, "meta.json")

    with open(meta_path, 'r') as f:
        meta = json.load(f)
        target_angle = float(meta['target_angle'])

    # Load and prep audio
    mixture = load_audio(mix_path).unsqueeze(0).to(DEVICE) # Add batch dim
    target = load_audio(target_path)
    if target.shape[0] > 1: target = target[0:1, :] # Force Mono target
    target = target.to(DEVICE)

    # Angle tensor
    angle_tensor = torch.tensor([target_angle], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 4. Run Inference
    with torch.no_grad():
        estimate = model(mixture, angle_tensor)
        
        # Align lengths for metrics (min length)
        min_len = min(estimate.shape[-1], target.shape[-1])
        est_trim = estimate[..., :min_len]
        tgt_trim = target[..., :min_len]

    # 5. Calculate Metrics
    print("\n[Calculating Metrics...]")
    
    # Initialize metric calculators
    # wb=True for WideBand PESQ (16k), False for NarrowBand (8k)
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=SAMPLE_RATE, mode='wb').to(DEVICE)
    stoi_metric = ShortTimeObjectiveIntelligibility(fs=SAMPLE_RATE, extended=False).to(DEVICE)
    sisdr_metric = ScaleInvariantSignalDistortionRatio().to(DEVICE)

    # Compute
    score_pesq = pesq_metric(est_trim, tgt_trim)
    score_stoi = stoi_metric(est_trim, tgt_trim)
    score_sisdr = sisdr_metric(est_trim, tgt_trim)

    print(f"SI-SDR: {score_sisdr.item():.4f} dB")
    print(f"STOI:   {score_stoi.item():.4f}")
    print(f"PESQ:   {score_pesq.item():.4f}")

    # 6. Save Results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sample_name = os.path.basename(SAMPLE_FOLDER)
    
    out_mix_path = os.path.join(OUTPUT_DIR, f"{sample_name}_mixture.wav")
    out_tgt_path = os.path.join(OUTPUT_DIR, f"{sample_name}_target.wav")
    out_est_path = os.path.join(OUTPUT_DIR, f"{sample_name}_output.wav")

    # Save audio (ensure on CPU for saving)
    # Mixture was (Batch=1, Channels=4, Time) -> squeeze to (4, Time)
    torchaudio.save(out_mix_path, mixture.squeeze(0).cpu(), SAMPLE_RATE)
    
    # Target was (1, Time) -> No change needed
    torchaudio.save(out_tgt_path, tgt_trim.cpu(), SAMPLE_RATE)
    
    # Estimate is (Batch=1, Time)
    # FIXED: Removed .squeeze(0) to keep shape as (1, Time) which torchaudio requires for mono
    torchaudio.save(out_est_path, est_trim.cpu(), SAMPLE_RATE)

    print(f"\n[Files Saved]")
    print(f"Mixture: {out_mix_path}")
    print(f"Target:  {out_tgt_path}")
    print(f"Output:  {out_est_path}")

if __name__ == "__main__":
    run_inference()