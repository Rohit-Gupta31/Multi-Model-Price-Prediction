"""
Calculate SMAPE score between predictions and ground truth
Matches by sample_id to ensure correct comparison
"""

import pandas as pd
import numpy as np


def compute_smape(y_true, y_pred):
    """
    Compute SMAPE (Symmetric Mean Absolute Percentage Error)
    This function expects ACTUAL PRICES (not log prices)
    
    Args:
        y_true: Ground truth prices (actual values)
        y_pred: Predicted prices (actual values)
    
    Returns:
        SMAPE score as percentage
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    
    # Use where to handle zero denominator (same as training code)
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    smape = np.mean(ratio) * 100
    
    return smape


def main():
    # Paths
    predictions_csv = "papersubmissiontrain.csv"  # Updated to match inference_mjave_exact.py output
    ground_truth_csv = "/media/KB/Segformer_ML/student_resource/dataset/train_filtered.csv"
    
    print("="*70)
    print("SMAPE Score Calculation")
    print("="*70)
    
    # Load both CSVs
    print(f"\nLoading predictions from: {predictions_csv}")
    pred_df = pd.read_csv(predictions_csv)
    print(f"  - Loaded {len(pred_df)} predictions")
    print(f"  - Columns: {list(pred_df.columns)}")
    
    print(f"\nLoading ground truth from: {ground_truth_csv}")
    truth_df = pd.read_csv(ground_truth_csv)
    print(f"  - Loaded {len(truth_df)} ground truth samples")
    print(f"  - Columns: {list(truth_df.columns)}")
    
    # Merge on sample_id to ensure matching
    print(f"\nMerging datasets on 'sample_id'...")
    merged_df = pred_df.merge(
        truth_df[['sample_id', 'price']], 
        on='sample_id', 
        suffixes=('_pred', '_true')
    )
    
    print(f"  - Matched {len(merged_df)} samples")
    
    if len(merged_df) == 0:
        print("❌ ERROR: No matching sample_ids found!")
        print("\nChecking sample_id formats:")
        print(f"  Predictions sample_ids (first 5): {pred_df['sample_id'].head().tolist()}")
        print(f"  Ground truth sample_ids (first 5): {truth_df['sample_id'].head().tolist()}")
        return
    
    # Check for missing matches
    missing_in_truth = len(pred_df) - len(merged_df)
    if missing_in_truth > 0:
        print(f"  ⚠️  Warning: {missing_in_truth} predictions had no matching ground truth")
    
    # Extract prices (these are ACTUAL prices from CSV)
    if 'price_pred' in merged_df.columns and 'price_true' in merged_df.columns:
        y_pred = merged_df['price_pred'].values
        y_true = merged_df['price_true'].values
    elif 'price' in pred_df.columns and 'price' in truth_df.columns:
        # If no suffix was added, manually extract
        y_pred = merged_df.iloc[:, merged_df.columns.get_loc('price')].values
        y_true = merged_df.iloc[:, merged_df.columns.get_loc('price') + 1].values
    else:
        print("❌ ERROR: Could not find price columns!")
        print(f"Merged columns: {list(merged_df.columns)}")
        return
    
    # Convert actual prices back to LOG SPACE (compute_smape expects log prices!)
    # CSV has actual prices, but compute_smape does expm1 internally
    y_pred_log = np.log1p(y_pred)
    y_true_log = np.log1p(y_true)
    
    # Calculate SMAPE
    print("\n" + "="*70)
    print("Calculating SMAPE...")
    print("="*70)
    
    smape_score = compute_smape(y_true_log, y_pred_log)
    
    print(f"\n✅ SMAPE Score: {smape_score:.4f}%")
    
    # Additional statistics
    print("\n" + "="*70)
    print("Additional Statistics")
    print("="*70)
    
    print(f"\nGround Truth Prices:")
    print(f"  - Min:    ${y_true.min():.2f}")
    print(f"  - Max:    ${y_true.max():.2f}")
    print(f"  - Mean:   ${y_true.mean():.2f}")
    print(f"  - Median: ${np.median(y_true):.2f}")
    
    print(f"\nPredicted Prices:")
    print(f"  - Min:    ${y_pred.min():.2f}")
    print(f"  - Max:    ${y_pred.max():.2f}")
    print(f"  - Mean:   ${y_pred.mean():.2f}")
    print(f"  - Median: ${np.median(y_pred):.2f}")
    
    print(f"\nErrors:")
    errors = np.abs(y_pred - y_true)
    print(f"  - Mean Absolute Error:  ${errors.mean():.2f}")
    print(f"  - Median Absolute Error: ${np.median(errors):.2f}")
    print(f"  - Max Error:            ${errors.max():.2f}")
    
    # Show some sample comparisons
    print("\n" + "="*70)
    print("Sample Comparisons (first 10):")
    print("="*70)
    print(f"{'Sample ID':<12} {'True Price':<12} {'Pred Price':<12} {'Error':<12} {'APE %':<10}")
    print("-"*70)
    for i in range(min(10, len(merged_df))):
        sample_id = merged_df.iloc[i]['sample_id']
        true_price = y_true[i]
        pred_price = y_pred[i]
        error = abs(pred_price - true_price)
        ape = (error / ((abs(true_price) + abs(pred_price)) / 2)) * 100
        print(f"{sample_id:<12} ${true_price:<11.2f} ${pred_price:<11.2f} ${error:<11.2f} {ape:<10.2f}%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
