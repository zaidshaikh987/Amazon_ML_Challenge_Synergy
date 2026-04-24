#!/usr/bin/env python3
"""
ULTRA-FAST async image downloader with aiohttp.
Downloads product images with 100+ concurrent connections for maximum speed.
"""

import os
import sys
import argparse
import pandas as pd
import asyncio
import aiohttp
from pathlib import Path
from urllib.parse import urlparse
from tqdm.asyncio import tqdm as async_tqdm
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Semaphore to limit concurrent connections
MAX_CONCURRENT = 100  # Increased for speed!

async def download_single_image_async(session, url, output_dir, sample_id, semaphore, max_retries=3, timeout=30):
    """
    Async download a single image with retry logic.
    
    Args:
        session: aiohttp ClientSession
        url: Image URL to download
        output_dir: Directory to save the image
        sample_id: Sample ID to use as filename
        semaphore: Asyncio semaphore for rate limiting
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
    
    Returns:
        dict: Download result
    """
    if not url or pd.isna(url):
        return {'sample_id': sample_id, 'url': url, 'success': False, 'filename': None, 'error': 'Empty URL'}
    
    # Use sample_id as filename
    filename = f"{sample_id}.jpg"
    output_path = os.path.join(output_dir, filename)
    
    # Skip if already exists
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return {'sample_id': sample_id, 'url': url, 'success': True, 'filename': filename, 'error': None}
    
    # Download with retries
    async with semaphore:
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                timeout_obj = aiohttp.ClientTimeout(total=timeout)
                async with session.get(url, headers=headers, timeout=timeout_obj) as response:
                    response.raise_for_status()
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if not content_type.startswith('image/'):
                        return {'sample_id': sample_id, 'url': url, 'success': False, 
                               'filename': filename, 'error': f'Not an image: {content_type}'}
                    
                    # Download and save
                    content = await response.read()
                    
                    if len(content) > 0:
                        with open(output_path, 'wb') as f:
                            f.write(content)
                        return {'sample_id': sample_id, 'url': url, 'success': True, 
                               'filename': filename, 'error': None}
                    else:
                        if os.path.exists(output_path):
                            os.remove(output_path)
                            
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                else:
                    return {'sample_id': sample_id, 'url': url, 'success': False, 
                           'filename': filename, 'error': 'Timeout'}
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                else:
                    return {'sample_id': sample_id, 'url': url, 'success': False, 
                           'filename': filename, 'error': str(e)}
        
        return {'sample_id': sample_id, 'url': url, 'success': False, 
               'filename': filename, 'error': 'Max retries exceeded'}

async def download_images_async(csv_path, output_dir, id_col='sample_id', url_col='image_link', 
                               max_concurrent=100, max_retries=3):
    """
    ASYNC: Download images from CSV with massive parallelization.
    
    Args:
        csv_path: Path to CSV file
        output_dir: Directory to save images
        id_col: Column name for sample IDs
        url_col: Column name for image URLs
        max_concurrent: Maximum concurrent connections (default 100)
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
    logger.info(f"Using {max_concurrent} concurrent connections for ULTRA-FAST download!")
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create connector with custom settings
    connector = aiohttp.TCPConnector(
        limit=max_concurrent,
        limit_per_host=50,
        ttl_dns_cache=300
    )
    
    # Download all images concurrently
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for _, row in valid_urls.iterrows():
            task = download_single_image_async(
                session,
                row[url_col],
                output_dir,
                row[id_col],
                semaphore,
                max_retries
            )
            tasks.append(task)
        
        # Execute with progress bar
        results = []
        for coro in async_tqdm.as_completed(tasks, desc="Downloading images", total=len(tasks)):
            result = await coro
            results.append(result)
    
    # Print statistics
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    failed_downloads = [(r['sample_id'], r['url'], r['error']) 
                       for r in results if not r['success']]
    
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

def download_images_from_csv(csv_path, output_dir, id_col='sample_id', url_col='image_link', 
                           max_workers=100, max_retries=3):
    """
    Synchronous wrapper for async download function.
    """
    return asyncio.run(download_images_async(
        csv_path, output_dir, id_col, url_col, max_workers, max_retries
    ))

def main():
    parser = argparse.ArgumentParser(description='ULTRA-FAST async image downloader')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--out_dir', required=True, help='Output directory for images')
    parser.add_argument('--id_col', default='sample_id', help='Column name for sample IDs')
    parser.add_argument('--url_col', default='image_link', help='Column name for image URLs')
    parser.add_argument('--workers', type=int, default=100, help='Concurrent connections (default: 100)')
    parser.add_argument('--retries', type=int, default=5, help='Maximum retry attempts (default: 5)')
    
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
