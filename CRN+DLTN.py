import os
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

# --- 1. ROBUST TENSORFLOW IMPORT ---
try:
    import tensorflow as tf
except ImportError as e:
    print("\n" + "!"*60)
    print("CRITICAL ERROR: TensorFlow is not installed or corrupted.")
    print(f"Error Details: {e}")
    print("-" * 60)
    print("FIX: Run these commands in your terminal:")
    print("   pip uninstall -y tensorflow tensorflow-intel tensorflow-estimator")
    print("   pip install tensorflow==2.15.0")
    print("!"*60 + "\n")
    raise e

from torchmetrics.audio import ShortTimeObjectiveIntelligibility, PerceptualEvaluationSpeechQuality, ScaleInvariantSignalDistortionRatio
from tqdm import tqdm

# ==========================================
# 2. CONFIGURATION
# ==========================================
CRN_MODEL_PATH = "CRN_Reverb_finetuned.pth"
DTLN_MODEL_1_PATH = "model_1.tflite" 
DTLN_MODEL_2_PATH = "model_2.tflite" 

TEST_DATASET_ROOT = r"D:\test_reverb"
OUTPUT_DIR = "evaluation_Hybrid_CRN_DTLN"
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
DEVICE = torch.device("cpu")

# ==========================================
# 3. SPATIAL CRN MODEL (PyTorch)
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
# 4. ROBUST DTLN WRAPPER (Auto-Detects Inputs)
# ==========================================
class DTLN_Split_Wrapper:
    def __init__(self, model_1_path, model_2_path):
        """
        Robust wrapper that finds input/output indices by shape instead of hardcoding.
        """
        self.block_len = 512
        self.block_shift = 128
        
        # --- Load Model 1 ---
        try:
            self.interpreter_1 = tf.lite.Interpreter(model_path=model_1_path)
            self.interpreter_1.allocate_tensors()
            self.in_det_1 = self._sort_io_details(self.interpreter_1.get_input_details())
            self.out_det_1 = self._sort_io_details(self.interpreter_1.get_output_details())
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_1_path}: {e}")
        
        # --- Load Model 2 ---
        try:
            self.interpreter_2 = tf.lite.Interpreter(model_path=model_2_path)
            self.interpreter_2.allocate_tensors()
            self.in_det_2 = self._sort_io_details(self.interpreter_2.get_input_details())
            self.out_det_2 = self._sort_io_details(self.interpreter_2.get_output_details())
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_2_path}: {e}")

        self.reset_states()

    def _sort_io_details(self, details_list):
        """
        Sorts details into a dictionary: {'main': idx, 'states': [idx1, idx2]}
        Based on standard DTLN shapes:
        - Main input/output: Shape (1, 512) or (1, 128)
        - States: Shape (1, 1, 128)
        """
        sorted_io = {'main': None, 'states': []}
        for det in details_list:
            shape = det['shape']
            # Heuristic: States usually have 3 dims (1,1,128), Main has 2 (1, 512)
            if len(shape) == 3 and shape[1] == 1:
                sorted_io['states'].append(det)
            else:
                sorted_io['main'] = det
        
        # Ensure we found them
        if sorted_io['main'] is None:
            # Fallback: The largest tensor is likely main
            sorted_io['main'] = max(details_list, key=lambda x: np.prod(x['shape']))
            sorted_io['states'] = [d for d in details_list if d != sorted_io['main']]
            
        return sorted_io

    def reset_states(self):
        # Init zero states based on discovered shapes
        self.states_1 = [np.zeros(d['shape'], dtype=np.float32) for d in self.in_det_1['states']]
        self.states_2 = [np.zeros(d['shape'], dtype=np.float32) for d in self.in_det_2['states']]

    def process(self, audio_np):
        self.reset_states()
        if audio_np.ndim > 1: audio_np = audio_np.flatten()
        
        num_blocks = (len(audio_np) - self.block_len) // self.block_shift
        output_audio = []
        
        in_buffer = np.zeros((1, self.block_len), dtype=np.float32)
        
        for i in range(num_blocks):
            idx = i * self.block_shift
            in_buffer[0] = audio_np[idx : idx + self.block_len]
            
            # --- RUN MODEL 1 ---
            # Set Main Input
            self.interpreter_1.set_tensor(self.in_det_1['main']['index'], in_buffer)
            # Set State Inputs
            for s_idx, state_det in enumerate(self.in_det_1['states']):
                self.interpreter_1.set_tensor(state_det['index'], self.states_1[s_idx])
            
            self.interpreter_1.invoke()
            
            # Get Outputs
            out_1 = self.interpreter_1.get_tensor(self.out_det_1['main']['index'])
            # Update States
            self.states_1 = [self.interpreter_1.get_tensor(d['index']) for d in self.out_det_1['states']]
            
            # --- RUN MODEL 2 ---
            # Set Main Input (Output of Model 1)
            self.interpreter_2.set_tensor(self.in_det_2['main']['index'], out_1)
            # Set State Inputs
            for s_idx, state_det in enumerate(self.in_det_2['states']):
                self.interpreter_2.set_tensor(state_det['index'], self.states_2[s_idx])
            
            self.interpreter_2.invoke()
            
            # Get Final Output
            out_2 = self.interpreter_2.get_tensor(self.out_det_2['main']['index'])
            # Update States
            self.states_2 = [self.interpreter_2.get_tensor(d['index']) for d in self.out_det_2['states']]
            
            output_audio.extend(out_2[0])
            
        return np.array(output_audio, dtype=np.float32)

# ==========================================
# 5. MAIN EVALUATION PIPELINE
# ==========================================
def run_evaluation():
    print(f"--- Running Hybrid Evaluation (CRN + Split-DTLN) on {DEVICE} ---")
    
    if not os.path.exists(DTLN_MODEL_1_PATH) or not os.path.exists(DTLN_MODEL_2_PATH):
        print("ERROR: model_1.tflite or model_2.tflite not found!")
        return

    # 1. Load CRN
    crn_model = SpatialMediumCRN(n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEVICE)
    try:
        state_dict = torch.load(CRN_MODEL_PATH, map_location=DEVICE, weights_only=True)
        crn_model.load_state_dict(state_dict)
        print(f"CRN Model loaded: {CRN_MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: CRN Model file '{CRN_MODEL_PATH}' not found!")
        return
    crn_model.eval()

    # 2. Load DTLN
    try:
        dtln_model = DTLN_Split_Wrapper(DTLN_MODEL_1_PATH, DTLN_MODEL_2_PATH)
        print(f"DTLN Models loaded successfully.")
    except Exception as e:
        print(f"Error loading DTLN Models: {e}")
        return

    # 3. Setup Metrics
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=SAMPLE_RATE, mode='wb').to(DEVICE)
    stoi_metric = ShortTimeObjectiveIntelligibility(fs=SAMPLE_RATE, extended=False).to(DEVICE)
    sisdr_metric = ScaleInvariantSignalDistortionRatio().to(DEVICE)

    # 4. Find Samples
    sample_folders = sorted(glob.glob(os.path.join(TEST_DATASET_ROOT, "*sample_*")))
    
    if len(sample_folders) == 0:
        print(f"No samples found in {TEST_DATASET_ROOT}")
        return

    print(f"Found {len(sample_folders)} samples. Starting processing...")
    
    results = {'sisdr': [], 'stoi': [], 'pesq': []}
    best_stoi = -1.0
    best_stoi_sample = ""
    best_stoi_stats = {}

    # 5. Processing Loop
    for folder_path in tqdm(sample_folders):
        sample_name = os.path.basename(folder_path)
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

            # Stage 1: CRN
            angle_tensor = torch.tensor([target_angle], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                est_crn = crn_model(mixture, angle_tensor)
                
            est_crn_np = est_crn.squeeze().cpu().numpy()

            # Stage 2: DTLN
            est_crn_np = np.pad(est_crn_np, (512, 512), 'constant')
            est_dtln_np = dtln_model.process(est_crn_np)
            
            estimate_final = torch.tensor(est_dtln_np, device=DEVICE).unsqueeze(0)

            # Metrics
            min_len = min(estimate_final.shape[-1], target.shape[-1])
            est_trim = estimate_final[..., :min_len]
            tgt_trim = target[..., :min_len]

            s_pesq = pesq_metric(est_trim, tgt_trim).item()
            s_stoi = stoi_metric(est_trim, tgt_trim).item()
            s_sisdr = sisdr_metric(est_trim, tgt_trim).item()
            
            results['pesq'].append(s_pesq)
            results['stoi'].append(s_stoi)
            results['sisdr'].append(s_sisdr)
            
            if s_stoi > best_stoi:
                best_stoi = s_stoi
                best_stoi_sample = sample_name
                best_stoi_stats = {'sisdr': s_sisdr, 'pesq': s_pesq, 'stoi': s_stoi}
                
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                torchaudio.save(os.path.join(OUTPUT_DIR, f"BEST_STOI_hybrid_out.wav"), est_trim.cpu(), SAMPLE_RATE)
                torchaudio.save(os.path.join(OUTPUT_DIR, f"BEST_STOI_crn_only.wav"), est_crn.cpu(), SAMPLE_RATE) 
                torchaudio.save(os.path.join(OUTPUT_DIR, f"BEST_STOI_target.wav"), tgt_trim.cpu(), SAMPLE_RATE)

        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
            continue

    if len(results['sisdr']) > 0:
        avg_sisdr = np.mean(results['sisdr'])
        avg_stoi = np.mean(results['stoi'])
        avg_pesq = np.mean(results['pesq'])

        print("\n" + "="*40)
        print("   HYBRID (CRN + SPLIT-DTLN) RESULTS")
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
        print("-" * 40)
    else:
        print("No samples were successfully processed.")

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

if __name__ == "__main__":
    run_evaluation()