#!/usr/bin/env python3
"""Check image download status and identify issues."""

import pandas as pd
import os
from pathlib import Path

print("=" * 70)
print("IMAGE DOWNLOAD DIAGNOSTICS")
print("=" * 70)

# Check sample images
print("\n### SAMPLE TEST IMAGES ###")
if os.path.exists('images/sample/download_log.csv'):
    dl = pd.read_csv('images/sample/download_log.csv')
    failed = dl[dl['status'] != 'downloaded']
    print(f'Total: {len(dl)}, Downloaded: {len(dl[dl["status"] == "downloaded"])}, Failed: {len(failed)}')
    
    if len(failed) > 0:
        print('\nFailed reasons:')
        print(failed['error'].value_counts())
        print('\nFirst 10 failed entries:')
        print(failed[['sample_id', 'image_link', 'error']].head(10))
else:
    print("No download log found!")

# Check small_train images  
print("\n### SMALL TRAIN IMAGES ###")
if os.path.exists('images/small_train/download_log.csv'):
    dl = pd.read_csv('images/small_train/download_log.csv')
    failed = dl[dl['status'] != 'downloaded']
    print(f'Total: {len(dl)}, Downloaded: {len(dl[dl["status"] == "downloaded"])}, Failed: {len(failed)}')
    
    if len(failed) > 0:
        print('\nFailed reasons:')
        print(failed['error'].value_counts())
        print('\nFirst 10 failed entries:')
        print(failed[['sample_id', 'image_link', 'error']].head(10))
else:
    print("No download log found!")

# Check actual files vs expected
print("\n### FILE EXISTENCE CHECK ###")
if os.path.exists('student_resource/dataset/sample_test.csv'):
    df = pd.read_csv('student_resource/dataset/sample_test.csv')
    image_files = set(os.listdir('images/sample')) if os.path.exists('images/sample') else set()
    
    missing_count = 0
    for sid in df['sample_id']:
        filename = f"{sid}.jpg"
        if filename not in image_files:
            missing_count += 1
    
    print(f"Sample test: {len(df)} expected, {len(image_files)} files in dir, {missing_count} missing")

if os.path.exists('student_resource/dataset/small_train.csv'):
    df = pd.read_csv('student_resource/dataset/small_train.csv')
    image_files = set(os.listdir('images/small_train')) if os.path.exists('images/small_train') else set()
    
    missing_count = 0
    for sid in df['sample_id']:
        filename = f"{sid}.jpg"
        if filename not in image_files:
            missing_count += 1
    
    print(f"Small train: {len(df)} expected, {len(image_files)} files in dir, {missing_count} missing")

print("\n" + "=" * 70)
