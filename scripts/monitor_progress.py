#!/usr/bin/env python3
"""
Monitor preprocessing progress by checking processed files.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def monitor_progress(output_dir: Path):
    """Monitor preprocessing progress."""

    # Load recommended subjects
    rec_file = Path("data/raw/recommended_subjects.json")
    if rec_file.exists():
        with open(rec_file) as f:
            rec_data = json.load(f)
            expected_subjects = rec_data.get('subject_ids', [])
            total_files = rec_data.get('total_files', 0)
    else:
        print("❌ Recommended subjects file not found")
        return

    # Check processed subjects
    processed_subjects = []
    total_utterances = 0

    for subject_id in expected_subjects:
        subject_dir = output_dir / subject_id
        if subject_dir.exists():
            summary_file = subject_dir / 'summary.json'
            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                    processed_subjects.append(subject_id)
                    total_utterances += summary.get('processed', 0)

    # Display progress
    print("\n" + "="*70)
    print("📊 PREPROCESSING PROGRESS")
    print("="*70)
    print(f"⏰ Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"📁 Subjects:    {len(processed_subjects)}/{len(expected_subjects)} completed")
    print(f"📄 Utterances:  {total_utterances}/{total_files} processed")
    print(f"📈 Progress:    {total_utterances/total_files*100:.1f}%")
    print()

    # Show completed subjects
    if processed_subjects:
        print("✅ Completed subjects:")
        for subj in processed_subjects:
            subject_dir = output_dir / subj
            summary_file = subject_dir / 'summary.json'
            with open(summary_file) as f:
                summary = json.load(f)
                processed = summary.get('processed', 0)
                total = summary.get('total_utterances', 0)
                print(f"   {subj}: {processed}/{total} utterances")

    # Show remaining subjects
    remaining = [s for s in expected_subjects if s not in processed_subjects]
    if remaining:
        print()
        print(f"⏳ Remaining: {', '.join(remaining[:5])}", end="")
        if len(remaining) > 5:
            print(f" ... (+{len(remaining)-5} more)")
        else:
            print()

    print("="*70)

    # Estimate time remaining
    if total_utterances > 0 and total_utterances < total_files:
        # Assume ~3.5 seconds per utterance
        remaining_utterances = total_files - total_utterances
        estimated_seconds = remaining_utterances * 3.5
        estimated_minutes = estimated_seconds / 60
        print(f"⏱️  Estimated time remaining: ~{estimated_minutes:.1f} minutes")
        print("="*70)


if __name__ == "__main__":
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/aligned")
    monitor_progress(output_dir)
