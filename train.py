import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from scipy.signal import hilbert, butter, filtfilt
import warnings
warnings.filterwarnings('ignore')
import os
import glob
import h5py

# ==================== EEG Data Augmentation Tools ====================
import random

def eeg_add_noise(x, sigma_scale=0.02):
    """Add Gaussian noise to EEG signals"""
    return x + np.random.randn(*x.shape) * (sigma_scale * x.std())

def eeg_time_shift(x, max_shift=50):
    """Apply time shifting to EEG signals"""
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(x, shift, axis=-1)

def eeg_channel_dropout(x, p=0.1):
    """Randomly dropout EEG channels (simulating bad electrodes)"""
    mask = (np.random.rand(x.shape[0]) > p).astype(float)[:,None]
    return x * mask

def eeg_augment(x):
    """Apply random EEG data augmentation"""
    if np.random.rand() < 0.5:
        x = eeg_add_noise(x, 0.02)
    if np.random.rand() < 0.5:
        x = eeg_time_shift(x, 40)
    if np.random.rand() < 0.2:
        x = eeg_channel_dropout(x, 0.1)
    return x


# ==================== 1. MAT Data Loader and Preprocessing ====================
class MatDataLoader:
    def __init__(self, data_path, fs=1000, target_fs=250):
        """
        MAT format data loader
        Expected data structure:
        movement_left: 68×71680 single
        imagery_left: 68×358400 single
        frame: [-2000,5000]
        """
        self.data_path = data_path
        self.original_fs = fs  # Can be overridden in load_file if different per file
        self.target_fs = target_fs  # Target sampling rate
        self.downsample_factor = (fs // target_fs) if (fs is not None and fs > target_fs) else 1
        
    def load_subject(self, subject_id):
        """Load MAT file for a single subject"""
        file_path = f"{self.data_path}/subject_{subject_id:02d}.mat"
        if not os.path.exists(file_path):
            alt = os.path.join(self.data_path, f"s{subject_id:02d}.mat")
            if os.path.exists(alt):
                file_path = alt
        data = sio.loadmat(file_path, simplify_cells=True)
        
        # Extract motor imagery data
        imagery_left = data['imagery_left']  # 68×358400
        imagery_right = data['imagery_right']  # 68×358400
        imagery_event = data['imagery_event']  # 1×358400
        
        # Get trial information
        frame = data['frame']  # [-2000, 5000]
        n_trials = data['n_imagery_trials']  # 100
        
        # Calculate single trial length
        total_samples_left = imagery_left.shape[1]
        trial_samples = total_samples_left // n_trials  # 358400/100=3584
        
        # Segment trials
        left_trials, left_labels = self.segment_trials(imagery_left, imagery_event, trial_samples, label=0)
        right_trials, right_labels = self.segment_trials(imagery_right, imagery_event, trial_samples, label=1)
        
        # Combine
        all_trials = np.concatenate([left_trials, right_trials], axis=0)
        all_labels = np.concatenate([left_labels, right_labels], axis=0)
        
        # Preprocess
        processed_trials = self.preprocess_trials(all_trials)
        
        # Get electrode positions
        electrode_positions = data['senloc'] if 'senloc' in data else None  # 64×3
        
        # Get bad trial information
        bad_trials = data.get('bad_trial_indices', {})
        
        return processed_trials, all_labels, electrode_positions, bad_trials

    def load_file(self, file_path):
        """Load MAT file for a single subject by file path"""
        needed = ['imagery_left', 'imagery_right', 'imagery_event']
        imagery_left = imagery_right = imagery_event = None
        frame = None
        n_trials = None

        try:
            if h5py.is_hdf5(file_path):
                with h5py.File(file_path, 'r') as f:
                    for name in needed:
                        if name in f:
                            imagery = np.array(f[name])
                            if name == 'imagery_left':
                                imagery_left = imagery
                            elif name == 'imagery_right':
                                imagery_right = imagery
                            elif name == 'imagery_event':
                                imagery_event = imagery
                    if 'frame' in f:
                        try:
                            frame = np.array(f['frame'])
                        except Exception:
                            frame = None
                    if 'n_imagery_trials' in f:
                        try:
                            n_trials = int(np.array(f['n_imagery_trials']).squeeze())
                        except Exception:
                            n_trials = None
            else:
                # For non-HDF5 files, list variables and load only needed fields
                try:
                    var_info = sio.whosmat(file_path)
                    var_names = [v[0] for v in var_info]
                except Exception:
                    var_names = []

                to_load = [n for n in needed + ['frame', 'n_imagery_trials'] if n in var_names]
                if len(to_load) > 0:
                    data = sio.loadmat(file_path, variable_names=to_load, simplify_cells=True)
                else:
                    data = sio.loadmat(file_path, simplify_cells=True)

                imagery_left = data.get('imagery_left')
                imagery_right = data.get('imagery_right')
                imagery_event = data.get('imagery_event')
                frame = data.get('frame')
                n_trials = data.get('n_imagery_trials')

        except Exception as e:
            raise RuntimeError(f"Failed to read {file_path}: {e}")

        # If fields not found directly, try extracting from 'eeg' structure
        if imagery_left is None or imagery_right is None or imagery_event is None:
            try:
                eeg_container = None
                if 'data' in locals() and isinstance(data, dict) and 'eeg' in data:
                    eeg_container = data['eeg']
                else:
                    try:
                        if h5py.is_hdf5(file_path):
                            with h5py.File(file_path, 'r') as f:
                                if 'eeg' in f:
                                    eeg_container = np.array(f['eeg'])
                    except Exception:
                        eeg_container = None

                if eeg_container is not None:
                    if isinstance(eeg_container, dict):
                        for name in needed:
                            if name in eeg_container and (name == 'imagery_event' or eeg_container[name] is not None):
                                try:
                                    val = eeg_container[name]
                                except Exception:
                                    val = None

                                if isinstance(val, np.ndarray) and val.dtype == object:
                                    flat = val.flatten()
                                    chosen = None
                                    for v in flat:
                                        if isinstance(v, np.ndarray) and v.size > 0:
                                            chosen = v
                                            break
                                    if chosen is None and flat.size > 0:
                                        chosen = flat[0]
                                    val = chosen

                                if name == 'imagery_left' and imagery_left is None:
                                    imagery_left = val
                                if name == 'imagery_right' and imagery_right is None:
                                    imagery_right = val
                                if name == 'imagery_event' and imagery_event is None:
                                    imagery_event = val
                        if 'senloc' in eeg_container:
                            try:
                                electrode_positions = np.array(eeg_container['senloc'])
                            except Exception:
                                electrode_positions = eeg_container.get('senloc')
                        if 'n_imagery_trials' in eeg_container:
                            try:
                                n_trials = int(np.array(eeg_container['n_imagery_trials']).squeeze())
                            except Exception:
                                n_trials = eeg_container.get('n_imagery_trials')
                        if imagery_left is not None and imagery_right is not None and imagery_event is not None:
                            pass
                        else:
                            pass
                    
                    try:
                        eeg_flat = np.asarray(eeg_container).flatten()
                        if eeg_flat.size > 0:
                            elem = eeg_flat[0]
                        else:
                            elem = eeg_container
                    except Exception:
                        elem = eeg_container

                    names = None
                    try:
                        names = getattr(getattr(elem, 'dtype', None), 'names', None)
                    except Exception:
                        names = None

                    if names is not None:
                        for name in needed:
                            if name in names:
                                try:
                                    val = elem[name]
                                except Exception:
                                    try:
                                        val = np.asarray(elem[name])
                                    except Exception:
                                        val = None

                                if isinstance(val, np.ndarray) and val.dtype == object:
                                    flat = val.flatten()
                                    chosen = None
                                    for v in flat:
                                        if isinstance(v, np.ndarray) and v.size > 0:
                                            chosen = v
                                            break
                                    if chosen is None and flat.size > 0:
                                        chosen = flat[0]
                                    val = chosen

                                if name == 'imagery_left' and imagery_left is None:
                                    imagery_left = val
                                if name == 'imagery_right' and imagery_right is None:
                                    imagery_right = val
                                if name == 'imagery_event' and imagery_event is None:
                                    imagery_event = val
            except Exception:
                pass

        # Final heuristic search for imagery/event arrays
        if imagery_left is None or imagery_right is None or imagery_event is None:
            try:
                found = self._search_for_imagery_and_events(file_path, locals().get('data', None))
                if found is not None:
                    il, ir, ie, nn, sen = found
                    if imagery_left is None:
                        imagery_left = il
                    if imagery_right is None:
                        imagery_right = ir
                    if imagery_event is None:
                        imagery_event = ie
                    if n_trials is None and nn is not None:
                        n_trials = nn
                    if electrode_positions is None and sen is not None:
                        electrode_positions = sen
            except Exception:
                pass

        if imagery_left is None or imagery_right is None or imagery_event is None:
            raise ValueError(f"Missing expected keys in {file_path}: found {', '.join([k for k,v in [('imagery_left',imagery_left),('imagery_right',imagery_right),('imagery_event',imagery_event)] if v is not None])}")

        # Try to read sampling rate from file
        try:
            file_srate = None
            if 'data' in locals() and isinstance(data, dict):
                if 'srate' in data:
                    file_srate = data['srate']
                elif 'eeg' in data and isinstance(data['eeg'], dict) and 'srate' in data['eeg']:
                    file_srate = data['eeg']['srate']
            if file_srate is None and h5py.is_hdf5(file_path):
                try:
                    with h5py.File(file_path, 'r') as f:
                        if 'srate' in f:
                            file_srate = np.array(f['srate'])
                except Exception:
                    file_srate = None

            if file_srate is not None:
                try:
                    fs_val = int(np.array(file_srate).squeeze())
                    self.original_fs = fs_val
                    self.downsample_factor = self.original_fs // self.target_fs if self.original_fs > self.target_fs else 1
                except Exception:
                    pass
        except Exception:
            pass

        total_samples_left = imagery_left.shape[1]
        trial_samples = total_samples_left // int(n_trials) if n_trials is not None else total_samples_left

        left_trials, left_labels = self.segment_trials(imagery_left, imagery_event, trial_samples, label=0)
        right_trials, right_labels = self.segment_trials(imagery_right, imagery_event, trial_samples, label=1)

        all_trials = np.concatenate([left_trials, right_trials], axis=0)
        all_labels = np.concatenate([left_labels, right_labels], axis=0)

        processed_trials = self.preprocess_trials(all_trials)

        electrode_positions = None
        try:
            if h5py.is_hdf5(file_path):
                with h5py.File(file_path, 'r') as f:
                    if 'senloc' in f:
                        electrode_positions = np.array(f['senloc'])
            else:
                if 'senloc' in data:
                    electrode_positions = data['senloc']
        except Exception:
            electrode_positions = None

        bad_trials = None
        if not h5py.is_hdf5(file_path):
            bad_trials = data.get('bad_trial_indices', {})

        return processed_trials, all_labels, electrode_positions, bad_trials
    
    def segment_trials(self, continuous_data, events, trial_samples, label):
        """Segment continuous data into trials"""
        event_indices = np.where(events != 0)[0]
        n_trials = len(event_indices)
        
        trials = np.zeros((n_trials, trial_samples, continuous_data.shape[0]))
        labels = np.ones(n_trials) * label
        
        for i, start_idx in enumerate(event_indices):
            if start_idx + trial_samples <= continuous_data.shape[1]:
                trial_data = continuous_data[:, start_idx:start_idx + trial_samples].T
                trials[i] = trial_data
        
        return trials, labels
    
    def preprocess_trials(self, trials):
        """Preprocess trial data"""
        processed = trials.copy()
        
        # 1. Downsampling
        if self.downsample_factor > 1:
            processed = processed[:, ::self.downsample_factor, :]
        
        # 2. Bandpass filtering (4-40 Hz)
        fs = self.target_fs
        nyquist = fs / 2
        low = 4 / nyquist
        high = 40 / nyquist
        b, a = butter(4, [low, high], btype='band')
        
        for i in range(processed.shape[0]):
            for j in range(processed.shape[2]):
                processed[i, :, j] = filtfilt(b, a, processed[i, :, j])
        
        # 3. Baseline correction (using first 200ms as baseline)
        baseline_samples = int(0.2 * fs)  # 200ms
        for i in range(processed.shape[0]):
            baseline = np.mean(processed[i, :baseline_samples, :], axis=0, keepdims=True)
            processed[i] = processed[i] - baseline
        
        # 4. Normalization (per channel)
        for i in range(processed.shape[2]):
            channel_data = processed[:, :, i]
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            if std_val > 0:
                processed[:, :, i] = (channel_data - mean_val) / std_val
        
        # Transpose to (n_trials, n_channels, n_samples) format
        processed = np.transpose(processed, (0, 2, 1))
        
        return processed

# ==================== 2. Core Neural Network Modules ====================
class SpectralEmbedding(nn.Module):
    """Spectral Convolution Embedding Module"""
    def __init__(self, in_channels, embed_dim, n_fft=64, hop_length=32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.conv = nn.Conv2d(1, embed_dim, kernel_size=(3, 3), padding=1)
        
    def stft_transform(self, x):
        """Compute STFT"""
        B, C, T = x.shape
        window = torch.hann_window(self.n_fft).to(x.device)

        specs = []
        for b in range(B):
            batch_specs = []
            for c in range(C):
                spec = torch.stft(x[b, c], n_fft=self.n_fft, hop_length=self.hop_length,
                                 window=window, return_complex=True)
                batch_specs.append(spec)
            specs.append(torch.stack(batch_specs))

        spec_tensor = torch.stack(specs)

        if not torch.is_tensor(spec_tensor):
            spec_tensor = torch.tensor(spec_tensor)

        spec_tensor = torch.abs(spec_tensor)
        if spec_tensor.dtype != torch.float32:
            spec_tensor = spec_tensor.float()

        return spec_tensor.unsqueeze(2)  # (B, C, 1, freq_dim, t_dim)

    def forward(self, x):
        # x: (B, C, T)
        spec = self.stft_transform(x)  # (B, C, 1, freq_dim, t_dim)

        B, C, _, freq_dim, t_dim = spec.shape
        spec = spec.reshape(B * C, 1, freq_dim, t_dim)

        if spec.dtype != torch.float32:
            spec = spec.float()

        embedded = self.conv(spec)  # (B*C, embed_dim, F', T')

        embedded = F.adaptive_avg_pool2d(embedded, (1, 1))
        embedded = embedded.reshape(B, C, -1)  # (B, C, embed_dim)

        return embedded

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Mask support (optional, currently not used)
        # if mask is not None:
        #     attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.out_proj(out)
        out = self.proj_dropout(out)

        return out

class GraphBiasedAttention(nn.Module):
    """Graph-Biased Attention Module"""
    def __init__(self, embed_dim, num_heads, electrode_positions=None, sigma=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Build spatial adjacency matrix
        if electrode_positions is not None:
            self.register_buffer('adjacency', self.build_adjacency(electrode_positions, sigma))
        else:
            self.adjacency = None
            
        self.lambda_scale = nn.Parameter(torch.tensor(0.1))
        
    def build_adjacency(self, positions, sigma=0.1):
        """Build adjacency matrix based on electrode positions"""
        C = positions.shape[0]
        positions_tensor = torch.tensor(positions, dtype=torch.float32)
        dist = torch.cdist(positions_tensor, positions_tensor)  # Euclidean distance
        adj = torch.exp(-dist**2 / (2 * sigma**2))
        return adj  # (C, C)
        
    def forward(self, x):
        # x: (B, C, D)
        B, C, D = x.shape
        qkv = self.qkv(x).reshape(B, C, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, C, D_h)
        
        # Standard attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, C, C)
        
        # Inject graph bias (if electrode positions available)
        if self.adjacency is not None:
            adj_expanded = self.adjacency.unsqueeze(0).unsqueeze(0)  # (1, 1, C, C)
            attn = attn + self.lambda_scale * adj_expanded
        
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, C, D)
        out = self.proj(out)
        return out, attn

class TransformerLayerWithAdapter(nn.Module):
    def __init__(self, embed_dim, num_heads, electrode_positions=None):
        super(TransformerLayerWithAdapter, self).__init__()
        
        ff_dim = embed_dim * 4
        dropout = 0.1
        
        # 1. Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(embed_dim)
        
        # 2. Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(embed_dim)
        
        # 3. Adapter (for domain adaptation)
        adapter_dim = max(8, embed_dim // 8)  # Ensure adapter dimension >= 8
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, embed_dim)
        )
        
        # 4. Electrode position encoding
        if electrode_positions is not None:
            self.electrode_encoding = nn.Parameter(torch.randn(len(electrode_positions), embed_dim))
        else:
            self.register_buffer('electrode_encoding', None)
    
    def forward(self, x, mask=None):
        # Add electrode position encoding
        if self.electrode_encoding is not None and x.shape[1] == self.electrode_encoding.shape[0]:
            x = x + self.electrode_encoding.unsqueeze(0)
        
        # Self-attention module
        residual = x
        x = self.attention_norm(x)
        attn_output = self.attention(x, mask)
        attn_output = self.attention_dropout(attn_output)
        x = residual + attn_output
        
        # Feed-forward module
        residual = x
        x = self.ff_norm(x)
        ff_output = self.ff(x)
        ff_output = self.ff_dropout(ff_output)
        x = residual + ff_output
        
        # Adapter module
        adapter_output = self.adapter(x)
        x = x + adapter_output
        
        return x, None

# ==================== 3. MotorFormer Model ====================
class MotorFormer(nn.Module):
    """Complete MotorFormer Model"""
    def __init__(self, n_channels, n_times, embed_dim=128, num_layers=6, 
                 num_heads=8, patch_size=10, electrode_positions=None):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Electrode positions
        self.electrode_positions = electrode_positions
        
        # Temporal embedding
        self.time_embed = nn.Conv1d(n_channels, embed_dim // 2, 
                                   kernel_size=patch_size, stride=patch_size)
        
        # Spectral embedding
        self.spectral_embed = SpectralEmbedding(n_channels, embed_dim // 2)
        
        # Position encoding
        self.num_patches = n_times // patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim // 2) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayerWithAdapter(embed_dim, num_heads, electrode_positions)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 2)  # Binary classification
        )
        
    def forward(self, x, mask_ratio=0.0, return_attn=False):
        # x: (B, C, T)
        B, C, T = x.shape
        
        # ===== Embeddings =====
        # Temporal embedding
        time_emb = self.time_embed(x)  # (B, D/2, num_patches)
        time_emb = time_emb.transpose(1, 2)  # (B, num_patches, D/2)
        
        # Add position encoding
        time_emb = time_emb + self.pos_embed
        
        # Spectral embedding
        spec_emb = self.spectral_embed(x)  # (B, C, D/2)
        spec_emb = spec_emb.mean(dim=1, keepdim=True)  # Average to each patch
        
        # Expand spectral embedding to all patches
        spec_emb_expanded = spec_emb.expand(-1, time_emb.shape[1], -1)
        
        # Concatenate
        embeddings = torch.cat([time_emb, spec_emb_expanded], dim=-1)  # (B, num_patches, D)
        
        # ===== Transformer Encoding =====
        all_attn = []
        for layer in self.layers:
            embeddings, attn = layer(embeddings)
            all_attn.append(attn)
        
        # Global average pooling
        pooled = embeddings.mean(dim=1)  # (B, D)
        
        # ===== Output =====
        logits = self.classifier(pooled)
        
        if return_attn:
            return logits, all_attn
        return logits

# ==================== 4. Training and Evaluation ====================
class Trainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

    def finetune(self, train_loader, val_loader, epochs=100, lr=1e-4, early_stop_patience=15):
        """Fine-tuning with EarlyStopping and data augmentation"""
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0
        best_model_state = None
        patience = 0
        for epoch in range(epochs):
            self.model.train()
            train_loss, train_correct = 0, 0
            total_train = 0
            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                # Data augmentation (only for training set)
                data_np = data.cpu().numpy()
                for i in range(data_np.shape[0]):
                    data_np[i] = eeg_augment(data_np[i])
                data = torch.tensor(data_np, dtype=torch.float32).to(self.device)
                optimizer.zero_grad()
                logits = self.model(data, mask_ratio=0)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                train_correct += (logits.argmax(1) == labels).sum().item()
                total_train += data.size(0)
            # Validation
            self.model.eval()
            val_correct = 0
            total_val = 0
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    logits = self.model(data, mask_ratio=0)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    total_val += data.size(0)
            train_loss = train_loss / total_train
            train_acc = train_correct / total_train * 100
            val_acc = val_correct / total_val * 100
            scheduler.step()
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience = 0
                torch.save(best_model_state, f'best_model_epoch_{epoch+1}.pth')
            else:
                patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return best_acc

# ==================== 5. Main Program ====================
def main():
    # Configuration parameters
    config = {
        'data_path': os.path.abspath(os.path.join(os.path.dirname(__file__), 'EEG_data')),
        'n_subjects': 52,
        'n_channels': 68,        # According to your data: 68 channels
        'target_fs': 250,        # Target sampling rate
        'trial_duration': 7.0,   # According to frame: [-2000, 5000] = 7000ms = 7 seconds
        'embed_dim': 128,
        'num_layers': 6,
        'num_heads': 8,
        'patch_size': 25,        # 100ms patch (250Hz * 0.1s = 25 points)
        'batch_size': 32,
        'finetune_epochs': 100
    }
    
    # Calculate number of time points
    n_times = int(config['trial_duration'] * config['target_fs'])  # 7 * 250 = 1750
    
    # 1. Load data
    print("Loading data...")
    loader = MatDataLoader(config['data_path'], fs=1000, target_fs=config['target_fs'])
    
    all_subjects_data = []
    all_subjects_labels = []
    all_electrode_positions = []
    
    mat_files = sorted(glob.glob(os.path.join(config['data_path'], '*.mat')))
    if len(mat_files) == 0:
        print(f"No .mat files found in {config['data_path']}")

    for file_path in mat_files:
        try:
            trials, labels, electrode_positions, _ = loader.load_file(file_path)

            # Ensure correct data shape
            if trials.shape[1] == config['n_channels'] and trials.shape[2] >= n_times:
                trials = trials[:, :, :n_times]
                all_subjects_data.append(trials)
                all_subjects_labels.append(labels)
                all_electrode_positions.append(electrode_positions)
                print(f"Loaded {os.path.basename(file_path)}: {len(trials)} trials, shape {trials.shape}")
            else:
                print(f"{os.path.basename(file_path)} shape mismatch: {trials.shape}")

        except Exception as e:
            print(f"Error loading {os.path.basename(file_path)}: {e}")
    
    # Exit gracefully if no subjects loaded
    if len(all_subjects_labels) == 0:
        print(f"No subjects loaded from {config['data_path']}. Check .mat variable names and formats.")
        return
    
    # ========== Train/Validation/Test Split ========== #
    from sklearn.model_selection import StratifiedKFold, train_test_split
    all_acc = []
    subject_ids = []
    subject_performance = []
    for i, labels in enumerate(all_subjects_labels):
        subject_ids.append(i)
        baseline_acc = max(np.mean(labels == 0), np.mean(labels == 1))
        subject_performance.append(0 if baseline_acc < 0.6 else 1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (trainval_idx, test_idx) in enumerate(skf.split(subject_ids, subject_performance)):
        print(f"\n=== Fold {fold+1}/5 ===")
        # First split: train+validation vs test
        trainval_data = [all_subjects_data[idx] for idx in trainval_idx]
        trainval_labels = [all_subjects_labels[idx] for idx in trainval_idx]
        test_data = [all_subjects_data[idx] for idx in test_idx]
        test_labels = [all_subjects_labels[idx] for idx in test_idx]
        trainval_data_tensor = np.concatenate(trainval_data, axis=0)
        trainval_labels_tensor = np.concatenate(trainval_labels, axis=0)
        # Second split: train vs validation within trainval
        train_idx, val_idx = train_test_split(np.arange(len(trainval_labels_tensor)), test_size=0.2, stratify=trainval_labels_tensor, random_state=fold)
        train_data_tensor = torch.tensor(trainval_data_tensor[train_idx], dtype=torch.float32)
        train_labels_tensor = torch.tensor(trainval_labels_tensor[train_idx], dtype=torch.long)
        val_data_tensor = torch.tensor(trainval_data_tensor[val_idx], dtype=torch.float32)
        val_labels_tensor = torch.tensor(trainval_labels_tensor[val_idx], dtype=torch.long)
        test_data_tensor = torch.tensor(np.concatenate(test_data, axis=0), dtype=torch.float32)
        test_labels_tensor = torch.tensor(np.concatenate(test_labels, axis=0), dtype=torch.long)
        # Electrode positions
        electrode_positions = None
        for idx in trainval_idx:
            if all_electrode_positions[idx] is not None:
                electrode_positions = all_electrode_positions[idx]
                break
        print(f"Creating model with {config['n_channels']} channels, {n_times} time points")
        model = MotorFormer(
            n_channels=config['n_channels'],
            n_times=n_times,
            embed_dim=config['embed_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            patch_size=config['patch_size'],
            electrode_positions=electrode_positions
        )
        trainer = Trainer(model, device='cuda' if torch.cuda.is_available() else 'cpu')
        # Data loaders
        train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_labels_tensor)
        val_dataset = torch.utils.data.TensorDataset(val_data_tensor, val_labels_tensor)
        test_dataset = torch.utils.data.TensorDataset(test_data_tensor, test_labels_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config['batch_size'], shuffle=False)
        # Training + Validation
        print("Finetuning...")
        val_acc = trainer.finetune(train_loader, val_loader, epochs=config['finetune_epochs'])
        # Test set evaluation
        model.eval()
        test_correct = 0
        total_test = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(trainer.device), labels.to(trainer.device)
                logits = model(data, mask_ratio=0)
                test_correct += (logits.argmax(1) == labels).sum().item()
                total_test += data.size(0)
        test_acc = test_correct / total_test * 100
        all_acc.append(test_acc)
        print(f"Fold {fold+1} test accuracy: {test_acc:.2f}%")
    print(f"\n=== Final Results ===")
    print(f"Mean Test Accuracy: {np.mean(all_acc):.2f}% ± {np.std(all_acc):.2f}%")
    print(f"Individual folds: {[f'{acc:.2f}%' for acc in all_acc]}")
    results = {
        'mean_test_accuracy': np.mean(all_acc),
        'std_test_accuracy': np.std(all_acc),
        'fold_accuracies': all_acc,
        'config': config
    }
    np.save('experiment_results.npy', results)
    print("Results saved to experiment_results.npy")

if __name__ == "__main__":
    main()