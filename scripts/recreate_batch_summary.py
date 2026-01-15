import json
from pathlib import Path

aligned_dir = Path('data/processed/aligned')
subjects = []

for subject_dir in sorted(aligned_dir.iterdir()):
    if not subject_dir.is_dir() or not subject_dir.name.startswith('sub'):
        continue
    
    subject_id = subject_dir.name
    utterances = []
    
    # Each utterance has a metadata JSON
    for meta_file in sorted(subject_dir.glob('*_metadata.json')):
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        
        # Get utterance name from filename or metadata
        utterance_name = meta_file.name.replace('_metadata.json', '')
        
        utterances.append({
            'utterance_name': utterance_name,
            'hdf5_path': str(aligned_dir / subject_id / f"{utterance_name}.h5"),
            'metadata_path': str(meta_file),
            'alignment_valid': meta.get('alignment', {}).get('is_valid', False),
            'correlation': meta.get('alignment', {}).get('correlation', 0.0)
        })
    
    subjects.append({
        'subject_id': subject_id,
        'total_utterances': len(utterances),
        'processed': len(utterances),
        'failed': 0,
        'utterances': utterances,
        'failed_utterances': []
    })

full_summary = {
    'total_subjects': len(subjects),
    'processed_subjects': len(subjects),
    'total_utterances': sum(s['total_utterances'] for s in subjects),
    'failed_utterances': 0,
    'subjects': subjects
}

with open(aligned_dir / 'batch_summary.json', 'w') as f:
    json.dump(full_summary, f, indent=2)

print(f"Recreated batch_summary.json with {len(subjects)} subjects and {full_summary['total_utterances']} utterances.")
