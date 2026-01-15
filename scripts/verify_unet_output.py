import torch
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

from src.segmentation.unet_simple import UNet

def verify_segmentation(subject_id, utterance_name):
    device = torch.device('cpu')
    model = UNet(n_channels=1, n_classes=1)
    
    # Load model
    model_path = 'models/unet_scratch/unet_best.pth'
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Load aligned data
    hdf5_path = f'data/processed/aligned/{subject_id}/{utterance_name}.h5'
    with h5py.File(hdf5_path, 'r') as f:
        mri_frames = f['mri_frames'][:5] # Just first 5 frames
    
    output_dir = Path('visualizations/verify_unet')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, frame in enumerate(mri_frames):
        # Normalize
        frame_norm = (frame - frame.mean()) / (frame.std() + 1e-8)
        
        # Pad to 96x96
        h, w = frame.shape
        pad_h = (96 - h) // 2
        pad_w = (96 - w) // 2
        frame_padded = np.pad(frame_norm, ((pad_h, 96-h-pad_h), (pad_w, 96-w-pad_w)), mode='constant')
        
        # Inference
        frame_tensor = torch.FloatTensor(frame_padded).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = model(frame_tensor)
            pred = (torch.sigmoid(output) > 0.5).int().squeeze().numpy()
        
        # Unpad
        seg = pred[pad_h:pad_h+h, pad_w:pad_w+w]
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(frame, cmap='gray')
        axes[0].set_title('Original MRI')
        axes[1].imshow(seg, cmap='jet')
        axes[1].set_title('Airway Segmentation')
        plt.savefig(output_dir / f'{utterance_name}_frame{i}.png')
        plt.close()

    print(f"Saved verification images to {output_dir}")

if __name__ == "__main__":
    verify_segmentation('sub011', 'sub011_2drt_04_bvt_r1_video')
