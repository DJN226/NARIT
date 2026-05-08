import os
import numpy as np
import pandas as pd
import feets
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import warnings
import logging
import feets.preprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def calculate_mag_error(flux, flux_err):
    """
    Calculate magnitude errors for flux measurements
    
    Parameters:
    flux: array of flux values
    flux_err: array of flux error values
    
    Returns:
    mag_err: array of magnitude errors
    """
    mag_err = np.full_like(flux, 99.99, dtype=float)
    
    # Only calculate for positive flux values
    positive_mask = (flux > 0) & (flux_err > 0)
    if np.any(positive_mask):
        mag_err[positive_mask] = 2.5 / np.log(10) * np.abs(flux_err[positive_mask] / flux[positive_mask])
        
        # Cap very large errors at reasonable value (10 mag error)
        mag_err = np.where(mag_err > 10, 99.99, mag_err)
    
    return mag_err

def load_asassn_data(filepath):
    """
    Load ASASSN .dat file and return structured data with improved error handling
    
    Parameters:
    filepath: path to .dat file
    
    Returns:
    dict with time, magnitude, mag_error, flux, flux_error arrays
    """
    try:
        # Read the file with better error handling
        data = pd.read_csv(filepath, sep='\s+', skiprows=1, 
                        #   names=['HJD', 'MAG', 'MAG_ERR', 'FLUX', 'FLUX_ERR'],
                          dtype={'HJD': float, 'MAG': float, 'MAG_ERR': float, 
                                'FLUX': float, 'FLUX_ERR': float},
                          na_values=['nan', 'NaN', 'null', ''])
        
        # Handle MAG_ERR column which might contain '99.99' as string
        mag_err_processed = []
        for val in data['MAG_ERR']:
            if pd.isna(val) or val == '' or val == 'nan':
                mag_err_processed.append(99.99)
            elif isinstance(val, str) and val.strip() == '99.99':
                mag_err_processed.append(99.99)
            else:
                try:
                    mag_err_processed.append(float(val))
                except (ValueError, TypeError):
                    mag_err_processed.append(99.99)
        
        # Convert to numpy arrays with proper type checking
        try:
            time = pd.to_numeric(data['HJD'], errors='coerce').values
            magnitude = pd.to_numeric(data['MAG'], errors='coerce').values
            mag_error = np.array(mag_err_processed, dtype=float)
            flux = pd.to_numeric(data['FLUX'], errors='coerce').values
            flux_error = pd.to_numeric(data['FLUX_ERR'], errors='coerce').values
        except Exception as e:
            logger.error(f"Error converting data types in {filepath}: {str(e)}")
            return {'success': False, 'error': f'Data type conversion error: {str(e)}'}
        
        # Check for any NaN values and log warning
        nan_mask = np.isnan(time) | np.isnan(magnitude) | np.isnan(flux) | np.isnan(flux_error)
        if np.any(nan_mask):
            logger.warning(f"Found {np.sum(nan_mask)} NaN values in {filepath}, will be filtered out")
        
        return {
            'time': time,
            'magnitude': magnitude,
            'mag_error': mag_error,
            'flux': flux,
            'flux_error': flux_error,
            'success': True
        }
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        return {'success': False, 'error': str(e)}

def process_single_file(args):
    """
    Process a single .dat file for feature extraction with improved error handling
    
    Parameters:
    args: tuple of (filepath, class_label)
    
    Returns:
    dict with results
    """
    filepath, class_label = args
    filename = Path(filepath).stem
    
    try:
        # Loading
        df = pd.read_csv(filepath, sep='\s+', skiprows=1)
        time = df["HJD"].values
        magnitude = df["MAG"].values
        mag_error = df["MAG_ERR"].values
        flux = df["FLUX"].values
        flux_error = df["FLUX_ERR"].values
        
        # Check if file has 99.99 errors before preprocessing
        had_99_errors = np.any(mag_error == 99.99)
        
        # Calculate magnitude errors for 99.99 values
        mag_err_calculated = calculate_mag_error(flux, flux_error)
        
        # Replace 99.99 errors with calculated ones
        mag_err_final = np.where(mag_error == 99.99, mag_err_calculated, mag_error)
        
        # Filter out problematic points with improved validation
        # Check for finite values and positive flux
        finite_mask = (np.isfinite(time) & np.isfinite(magnitude) & 
                      np.isfinite(mag_err_final) & np.isfinite(flux) & 
                      np.isfinite(flux_error))
        
        positive_flux_mask = flux > 0
        valid_error_mask = mag_err_final != 99.99
        reasonable_error_mask = mag_err_final < 10.0  # Additional check for reasonable errors
        
        # Combine all masks
        valid_mask = (finite_mask & positive_flux_mask & 
                     valid_error_mask & reasonable_error_mask)
        
        if np.sum(valid_mask) < 10:  # Need at least 10 points for meaningful features
            return {
                'filename': filename,
                'class': class_label,
                'success': False,
                'error': f'Insufficient valid data points: {np.sum(valid_mask)}/10 required',
                'had_99_errors': had_99_errors,
                'total_points': len(time),
                'finite_points': np.sum(finite_mask),
                'positive_flux_points': np.sum(positive_flux_mask),
                'valid_error_points': np.sum(valid_error_mask)
            }
        
        time_clean = time[valid_mask]
        mag_clean = magnitude[valid_mask]
        err_clean = mag_err_final[valid_mask]
        
        # Additional validation before FEETS preprocessing
        if len(time_clean) == 0 or np.all(mag_clean == mag_clean[0]):
            return {
                'filename': filename,
                'class': class_label,
                'success': False,
                'error': 'No variation in cleaned magnitude data',
                'had_99_errors': had_99_errors
            }
        
        # # Apply FEETS preprocessing with better error handling
        # try:
        #     time_processed, mag_processed, err_processed = feets.preprocess.remove_noise(
        #         time=time_clean,
        #         magnitude=mag_clean,
        #         error=err_clean
        #     )
            
        #     # Check if preprocessing removed too many points
        #     if len(time_processed) < 5:
        #         logger.warning(f"Preprocessing removed too many points for {filename}, using cleaned data")
        #         time_processed, mag_processed, err_processed = time_clean, mag_clean, err_clean
                
        # except Exception as e:
        #     # If preprocessing fails, use cleaned data directly
        #     logger.warning(f"Preprocessing failed for {filename}: {str(e)}, using cleaned data")
        #     time_processed, mag_processed, err_processed = time_clean, mag_clean, err_clean
        
        # # Final check before feature extraction
        # if len(time_processed) < 5:
        #     return {
        #         'filename': filename,
        #         'class': class_label,
        #         'success': False,
        #         'error': 'Insufficient points after preprocessing',
        #         'had_99_errors': had_99_errors
        #     }
        
        # Skip FEETS preprocessing, use cleaned data directly
        time_processed, mag_processed, err_processed = time_clean, mag_clean, err_clean
        
        # Extract features with error handling
        try:
            fs = feets.FeatureSpace(data=["time", "magnitude", "error"])
            features, values = fs.extract(time=time_processed, magnitude=mag_processed, error=err_processed)
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {filename}: {str(e)}")
            return {
                'filename': filename,
                'class': class_label,
                'success': False,
                'error': f'Feature extraction failed: {str(e)}',
                'had_99_errors': had_99_errors
    }
        
        # Create result dictionary
        result = {
            'filename': filename,
            'class': class_label,
            'success': True,
            'had_99_errors': had_99_errors,
            'n_original_points': len(time),
            'n_valid_points': len(time_clean),
            'n_processed_points': len(time_processed)
        }
        
        # Add feature values with validation
        for feature, value in zip(features, values):
            # Check if value is valid (not NaN or infinite)
            if np.isfinite(value):
                result[feature] = value
            else:
                result[feature] = np.nan
                logger.warning(f"Invalid feature value for {feature} in {filename}: {value}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return {
            'filename': filename,
            'class': class_label,
            'success': False,
            'error': str(e),
            'had_99_errors': False
        }

def process_class_folder(base_folder, class_name, output_folder, n_workers=None):
    """
    Process all .dat files in a class folder
    
    Parameters:
    base_folder: path to vardlc folder
    class_name: name of the class folder (e.g., 'var_Cataclysmic')
    output_folder: path to output folder
    n_workers: number of worker processes (None for auto)
    """
    
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), 8)  # Use up to 8 cores
    
    class_folder = Path(base_folder) / class_name
    
    if not class_folder.exists():
        logger.error(f"Class folder {class_folder} does not exist!")
        return
    
    # Find all .dat files
    dat_files = list(class_folder.glob("*.dat"))
    
    if not dat_files:
        logger.warning(f"No .dat files found in {class_folder}")
        return
    
    logger.info(f"Processing {len(dat_files)} files from class '{class_name}' using {n_workers} workers")
    
    # Prepare arguments for parallel processing
    args_list = [(str(filepath), class_name) for filepath in dat_files]
    
    # Process files in parallel
    results = []
    files_with_99_errors = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_file, args): args[0] for args in args_list}
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_file), total=len(args_list), 
                          desc=f"Processing {class_name}"):
            try:
                result = future.result()
                results.append(result)
                
                # Track files with 99.99 errors
                if result.get('had_99_errors', False):
                    files_with_99_errors.append(result['filename'])
                    
            except Exception as e:
                filepath = future_to_file[future]
                logger.error(f"Failed to process {filepath}: {str(e)}")
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    logger.info(f"Successfully processed {len(successful_results)}/{len(results)} files")
    
    if failed_results:
        logger.warning(f"Failed to process {len(failed_results)} files")
        # Group failures by error type for better reporting
        error_summary = {}
        for failed in failed_results:
            error_type = failed.get('error', 'Unknown error')
            if error_type not in error_summary:
                error_summary[error_type] = []
            error_summary[error_type].append(failed['filename'])
        
        for error_type, filenames in error_summary.items():
            logger.warning(f"  {error_type}: {len(filenames)} files")
            if len(filenames) <= 3:  # Show filenames if few
                for fname in filenames:
                    logger.warning(f"    - {fname}")
    
    # Save results to CSV
    if successful_results:
        df = pd.DataFrame(successful_results)
        
        # Create output folder if it doesn't exist
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_filename = f"{class_name}_features.csv"
        csv_path = output_path / csv_filename
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved features to {csv_path}")
        
        # Save list of files with 99.99 errors
        if files_with_99_errors:
            txt_filename = f"{class_name}_files_with_99_errors.txt"
            txt_path = output_path / txt_filename
            with open(txt_path, 'w') as f:
                f.write(f"Files from {class_name} that had MAG_ERR = 99.99 before preprocessing:\n")
                f.write(f"Total: {len(files_with_99_errors)} files\n\n")
                for filename in sorted(files_with_99_errors):
                    f.write(f"{filename}\n")
            logger.info(f"Saved list of files with 99.99 errors to {txt_path}")
        
        # Save detailed processing summary
        summary_filename = f"{class_name}_processing_summary.txt"
        summary_path = output_path / summary_filename
        with open(summary_path, 'w') as f:
            f.write(f"Processing Summary for {class_name}\n")
            f.write(f"================================\n\n")
            f.write(f"Total files found: {len(dat_files)}\n")
            f.write(f"Successfully processed: {len(successful_results)}\n")
            f.write(f"Failed to process: {len(failed_results)}\n")
            f.write(f"Files with 99.99 errors: {len(files_with_99_errors)}\n\n")
            
            if successful_results:
                df_stats = pd.DataFrame(successful_results)
                f.write(f"Feature extraction statistics:\n")
                f.write(f"Average original points per file: {df_stats['n_original_points'].mean():.1f}\n")
                f.write(f"Average valid points per file: {df_stats['n_valid_points'].mean():.1f}\n")
                f.write(f"Average processed points per file: {df_stats['n_processed_points'].mean():.1f}\n\n")
                
            # Write detailed failure analysis
            if failed_results:
                f.write(f"Failure Analysis:\n")
                f.write(f"=================\n")
                error_summary = {}
                for failed in failed_results:
                    error_type = failed.get('error', 'Unknown error')
                    if error_type not in error_summary:
                        error_summary[error_type] = []
                    error_summary[error_type].append(failed['filename'])
                
                for error_type, filenames in error_summary.items():
                    f.write(f"\n{error_type}: {len(filenames)} files\n")
                    for fname in filenames[:10]:  # Show up to 10 examples
                        f.write(f"  - {fname}\n")
                    if len(filenames) > 10:
                        f.write(f"  ... and {len(filenames) - 10} more\n")
        
        logger.info(f"Saved processing summary to {summary_path}")
    
    else:
        logger.error(f"No files were successfully processed for class {class_name}")

def main():
    """
    Main function to run feature extraction
    """
    # Configuration
    BASE_FOLDER = "vardlc"  # Change this to your actual path
    OUTPUT_FOLDER = "Variable_Stars_Features"
    
    # Example usage for a single class
    CLASS_NAME = "var_RRLyrae"  # Change this to the class you want to process
    
    # Number of worker processes (adjust based on your system)
    # Use fewer workers if you have limited RAM
    N_WORKERS = min(multiprocessing.cpu_count(), 6)
    
    logger.info(f"Starting feature extraction for class: {CLASS_NAME}")
    logger.info(f"Base folder: {BASE_FOLDER}")
    logger.info(f"Output folder: {OUTPUT_FOLDER}")
    logger.info(f"Using {N_WORKERS} worker processes")
    
    # Process the class
    process_class_folder(BASE_FOLDER, CLASS_NAME, OUTPUT_FOLDER, N_WORKERS)
    
    logger.info("Feature extraction completed!")

def process_all_classes():
    """
    Function to process all classes automatically
    """
    BASE_FOLDER = "vardlc"
    OUTPUT_FOLDER = "Variable_Stars_Features"
    N_WORKERS = min(multiprocessing.cpu_count(), 6)
    
    base_path = Path(BASE_FOLDER)
    if not base_path.exists():
        logger.error(f"Base folder {BASE_FOLDER} does not exist!")
        return
    
    # Find all class folders (folders starting with 'var_')
    class_folders = [d for d in base_path.iterdir() 
                    if d.is_dir() and d.name.startswith('var_')]
    
    logger.info(f"Found {len(class_folders)} class folders to process")
    
    for class_folder in class_folders:
        class_name = class_folder.name
        logger.info(f"Processing class: {class_name}")
        process_class_folder(BASE_FOLDER, class_name, OUTPUT_FOLDER, N_WORKERS)
        logger.info(f"Completed class: {class_name}\n")

if __name__ == "__main__":
    # For processing a single class, use:
    main()
    
    # For processing all classes automatically, uncomment and use:
    # process_all_classes()