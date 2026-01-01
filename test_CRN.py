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

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "CRN_Model_FineTuned_CompressedPESQ.pth"       
TEST_DATASET_ROOT = r"D:\test_dataset" 
OUTPUT_DIR = "evaluation_CRN"    
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
    
    if target_len:
        if waveform.shape[-1] < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]))
        else:
            waveform = waveform[:, :target_len]
            
    return waveform

# ==========================================
# 4. MAIN EVALUATION LOOP
# ==========================================
def run_evaluation():
    print(f"--- Running Full Evaluation on {DEVICE} ---")
    
    # 1. Load Model
    model = SpatialMediumCRN(n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEVICE)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Model loaded: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        return
    model.eval()

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
    
    # Storage for Best STOI
    best_stoi = -1.0
    best_stoi_sample = ""
    best_stoi_stats = {}

    # 4. Loop
    for folder_path in tqdm(sample_folders):
        sample_name = os.path.basename(folder_path)
        mix_path = os.path.join(folder_path, "mixture.wav")
        target_path = os.path.join(folder_path, "target.wav")
        meta_path = os.path.join(folder_path, "meta.json")

        try:
            # Load Angle
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                target_angle = float(meta['target_angle'])

            # Load Audio
            mixture = load_audio(mix_path).unsqueeze(0).to(DEVICE)
            target = load_audio(target_path)
            if target.shape[0] > 1: target = target[0:1, :] 
            target = target.to(DEVICE)

            # Run Inference
            angle_tensor = torch.tensor([target_angle], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                estimate = model(mixture, angle_tensor)
                
                # Align lengths
                min_len = min(estimate.shape[-1], target.shape[-1])
                est_trim = estimate[..., :min_len]
                tgt_trim = target[..., :min_len]

                # Compute Metrics
                s_pesq = pesq_metric(est_trim, tgt_trim).item()
                s_stoi = stoi_metric(est_trim, tgt_trim).item()
                s_sisdr = sisdr_metric(est_trim, tgt_trim).item()
                
                # Add to Averages
                results['pesq'].append(s_pesq)
                results['stoi'].append(s_stoi)
                results['sisdr'].append(s_sisdr)
                
                # Check for New Best STOI
                if s_stoi > best_stoi:
                    best_stoi = s_stoi
                    best_stoi_sample = sample_name
                    best_stoi_stats = {'sisdr': s_sisdr, 'pesq': s_pesq, 'stoi': s_stoi}
                    
                    # Save "BEST" files immediately
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    torchaudio.save(os.path.join(OUTPUT_DIR, f"BEST_STOI_output.wav"), est_trim.cpu(), SAMPLE_RATE)
                    torchaudio.save(os.path.join(OUTPUT_DIR, f"BEST_STOI_mixture.wav"), mixture.squeeze(0).cpu(), SAMPLE_RATE)
                    torchaudio.save(os.path.join(OUTPUT_DIR, f"BEST_STOI_target.wav"), tgt_trim.cpu(), SAMPLE_RATE)

        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
            continue

    # 5. Report Final Results
    avg_sisdr = np.mean(results['sisdr'])
    avg_stoi = np.mean(results['stoi'])
    avg_pesq = np.mean(results['pesq'])

    print("\n" + "="*40)
    print("   FINAL EVALUATION REPORT")
    print("="*40)
    print(f"Total Samples:   {len(results['sisdr'])}")
    print("-" * 40)
    print(f"AVERAGE SI-SDR:  {avg_sisdr:.4f} dB")
    print(f"AVERAGE STOI:    {avg_stoi:.4f}")
    print(f"AVERAGE PESQ:    {avg_pesq:.4f}")
    print("="*40)
    print("   HIGHEST STOI CASE")
    print("="*40)
    print(f"Sample:          {best_stoi_sample}")
    print(f"Best STOI:       {best_stoi:.4f}")
    print(f"Corresponding PESQ: {best_stoi_stats.get('pesq',0):.4f}")
    print(f"Corresponding SI-SDR: {best_stoi_stats.get('sisdr',0):.4f} dB")
    print("-" * 40)
    print(f"Best Audio Saved as 'BEST_STOI_output.wav' in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_evaluation()