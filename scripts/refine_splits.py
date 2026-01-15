import json
from pathlib import Path

mel_dir = Path('data/processed/audio_features/mel_spectrogram')
param_dir = Path('data/processed/parameters/geometric')
splits_dir = Path('data/processed/splits')

# Find utterances with both features
mel_files = {f.name.replace('_mel.npy', '') for f in mel_dir.glob('*.npy')}
param_files = {f.name.replace('_params.npy', '') for f in param_dir.glob('*.npy')}

available_utterances = mel_files.intersection(param_files)
print(f"Available utterances with both features: {len(available_utterances)}")

# Load original subject split
with open(splits_dir / 'split_info.json', 'r') as f:
    split_info = json.load(f)

new_splits = {}
for split_name in ['train', 'val', 'test']:
    subjects = set(split_info['splits'][split_name]['subjects'])
    utterances = [u for u in available_utterances if u.split('_')[0] in subjects]
    new_splits[split_name] = sorted(utterances)
    print(f"Split {split_name}: {len(utterances)} utterances")

# Save as JSON for create_dataloaders
for split_name in ['train', 'val', 'test']:
    with open(splits_dir / f'{split_name}.json', 'w') as f:
        json.dump(new_splits[split_name], f, indent=2)

print("Final splits saved to data/processed/splits/*.json")
