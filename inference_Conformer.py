"""
DCCRN-Conformer Inference Script
Usage: python inference_Conformer.py --input mixture.wav --angle 45 --output output.wav
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math

# ==========================================
# MODEL COMPONENTS (same as training)
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


class DCCRNConformer(nn.Module):
    def __init__(self, n_fft=512, hop_length=128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))
        
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
        
        self.angle_net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        self.conformer = DualPathConformer(256, num_blocks=3, num_heads=4, dropout=0.1)
        
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
        
        rad = torch.deg2rad(angle)
        angle_vec = torch.cat([torch.sin(rad), torch.cos(rad)], dim=1)
        angle_emb = self.angle_net(angle_vec).unsqueeze(-1).unsqueeze(-1)
        
        e4_r = e4_r + angle_emb
        e4_i = e4_i + angle_emb
        
        combined = torch.sqrt(e4_r**2 + e4_i**2 + 1e-8)
        combined = self.conformer(combined)
        
        e4_r = e4_r * combined
        e4_i = e4_i * combined
        
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
# INFERENCE - FIXED LENGTH
# ==========================================
SAMPLE_RATE = 16000
FIXED_DURATION = 3.0  # seconds - model's expected input size
FIXED_SAMPLES = int(SAMPLE_RATE * FIXED_DURATION)

def load_audio(path, sample_rate=16000):
    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    # Ensure stereo (2 channels)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]
    return waveform


def main():
    parser = argparse.ArgumentParser(description="DCCRN-Conformer Inference")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input audio file (stereo)")
    parser.add_argument("--angle", "-a", type=float, required=True, help="Target angle (0-180 degrees)")
    parser.add_argument("--output", "-o", type=str, default="output.wav", help="Output audio file")
    parser.add_argument("--model", "-m", type=str, default="DCCRN_Conformer.pth", help="Model checkpoint path")
    parser.add_argument("--device", "-d", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Load model
    model = DCCRNConformer(n_fft=512, hop_length=128).to(device)
    try:
        state_dict = torch.load(args.model, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Model loaded: {args.model}")
    except FileNotFoundError:
        print(f"Error: Model file '{args.model}' not found!")
        return
    model.eval()
    
    # Load audio
    print(f"Loading: {args.input}")
    waveform = load_audio(args.input)
    original_len = waveform.shape[-1]
    print(f"Input duration: {original_len / SAMPLE_RATE:.2f}s")
    
    # Cut or pad to fixed size (3 seconds)
    if original_len > FIXED_SAMPLES:
        waveform = waveform[:, :FIXED_SAMPLES]
        print(f"Trimmed to: {FIXED_DURATION}s")
    elif original_len < FIXED_SAMPLES:
        waveform = F.pad(waveform, (0, FIXED_SAMPLES - original_len))
        print(f"Padded to: {FIXED_DURATION}s")
    
    # Calculate input power (RMS) before processing
    input_rms = torch.sqrt(torch.mean(waveform ** 2))
    
    waveform = waveform.unsqueeze(0).to(device)  # [1, 2, T]
    
    # Create angle tensor
    angle_tensor = torch.tensor([[args.angle]], dtype=torch.float32).to(device)
    
    # Inference
    print(f"Processing with target angle: {args.angle}°")
    with torch.no_grad():
        output = model(waveform, angle_tensor)
    
    # Trim to original length if padded
    output_len = min(original_len, FIXED_SAMPLES)
    output = output[:, :output_len]
    
    # Match output power to input power
    output_rms = torch.sqrt(torch.mean(output.cpu() ** 2))
    if output_rms > 1e-8:
        scale = input_rms / output_rms
        output = output.cpu() * scale
        print(f"Power matched: scale={scale:.3f}")
    else:
        output = output.cpu()
    
    torchaudio.save(args.output, output, SAMPLE_RATE)
    print(f"Saved: {args.output}")
    print(f"Output duration: {output.shape[-1] / SAMPLE_RATE:.2f}s")


if __name__ == "__main__":
    main()

