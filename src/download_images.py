#!/usr/bin/env python3
"""
Robust image downloader with retries and error handling.
Downloads product images from URLs with multiprocessing support.
"""

import os
import sys
import argparse
import pandas as pd
import requests
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_single_image(url, output_dir, max_retries=3, timeout=30):
    """
    Download a single image with retry logic.
    
    Args:
        url: Image URL to download
        output_dir: Directory to save the image
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
    
    Returns:
        tuple: (success: bool, filename: str, error: str)
    """
    if not url or pd.isna(url):
        return False, None, "Empty URL"
    
    try:
        # Parse URL to get filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # If no filename, generate one
        if not filename or '.' not in filename:
            filename = f"image_{hash(url) % 100000}.jpg"
        
        output_path = os.path.join(output_dir, filename)
        
        # Skip if already exists
        if os.path.exists(output_path):
            return True, filename, None
        
        # Download with retries
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=timeout, stream=True)
                response.raise_for_status()
                
                # Check if it's actually an image
                content_type = response.headers.get('content-type', '').lower()
                if not content_type.startswith('image/'):
                    logger.warning(f"URL {url} doesn't return an image (content-type: {content_type})")
                    return False, filename, f"Not an image: {content_type}"
                
                # Save image
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify file was created and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return True, filename, None
                else:
                    logger.warning(f"Downloaded file is empty: {url}")
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return False, filename, str(e)
        
        return False, filename, "Max retries exceeded"
        
    except Exception as e:
        return False, None, str(e)

def download_images_from_csv(csv_path, output_dir, id_col='sample_id', url_col='image_link', 
                           max_workers=8, max_retries=3):
    """
    Download images from a CSV file containing image URLs.
    
    Args:
        csv_path: Path to CSV file
        output_dir: Directory to save images
        id_col: Column name for sample IDs
        url_col: Column name for image URLs
        max_workers: Number of concurrent download threads
        max_retries: Maximum retry attempts per image
    
    Returns:
        dict: Download statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV
    logger.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate columns
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in CSV")
    if url_col not in df.columns:
        raise ValueError(f"Column '{url_col}' not found in CSV")
    
    logger.info(f"Found {len(df)} rows in CSV")
    
    # Filter out empty URLs
    valid_urls = df[df[url_col].notna() & (df[url_col] != '')]
    logger.info(f"Found {len(valid_urls)} valid image URLs")
    
    # Download images
    results = []
    failed_downloads = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_url = {}
        for _, row in valid_urls.iterrows():
            future = executor.submit(
                download_single_image, 
                row[url_col], 
                output_dir, 
                max_retries
            )
            future_to_url[future] = (row[id_col], row[url_col])
        
        # Process completed downloads
        for future in tqdm(as_completed(future_to_url), total=len(future_to_url), desc="Downloading images"):
            sample_id, url = future_to_url[future]
            try:
                success, filename, error = future.result()
                results.append({
                    'sample_id': sample_id,
                    'url': url,
                    'success': success,
                    'filename': filename,
                    'error': error
                })
                
                if not success:
                    failed_downloads.append((sample_id, url, error))
                    
            except Exception as e:
                logger.error(f"Unexpected error for {url}: {str(e)}")
                failed_downloads.append((sample_id, url, str(e)))
    
    # Print statistics
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    logger.info(f"Download completed: {successful}/{total} images downloaded successfully")
    
    if failed_downloads:
        logger.warning(f"Failed downloads: {len(failed_downloads)}")
        logger.warning("First 5 failed downloads:")
        for i, (sample_id, url, error) in enumerate(failed_downloads[:5]):
            logger.warning(f"  {sample_id}: {url} - {error}")
    
    # Save download log
    results_df = pd.DataFrame(results)
    log_path = os.path.join(output_dir, 'download_log.csv')
    results_df.to_csv(log_path, index=False)
    logger.info(f"Download log saved to {log_path}")
    
    return {
        'total': total,
        'successful': successful,
        'failed': len(failed_downloads),
        'failed_downloads': failed_downloads,
        'results_df': results_df
    }

def main():
    parser = argparse.ArgumentParser(description='Download product images from CSV')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--out_dir', required=True, help='Output directory for images')
    parser.add_argument('--id_col', default='sample_id', help='Column name for sample IDs')
    parser.add_argument('--url_col', default='image_link', help='Column name for image URLs')
    parser.add_argument('--workers', type=int, default=8, help='Number of concurrent workers')
    parser.add_argument('--retries', type=int, default=3, help='Maximum retry attempts')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    try:
        stats = download_images_from_csv(
            csv_path=args.input,
            output_dir=args.out_dir,
            id_col=args.id_col,
            url_col=args.url_col,
            max_workers=args.workers,
            max_retries=args.retries
        )
        
        logger.info("Image download completed successfully!")
        logger.info(f"Statistics: {stats['successful']}/{stats['total']} images downloaded")
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
