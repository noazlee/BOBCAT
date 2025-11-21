import torch
import numpy as np
from model import MAMLModel

def load_and_inspect_model(model_path, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"{'='*80}")
    
    print("\nCheckpoint keys:", checkpoint.keys())
    print("\nModel state dict keys:", checkpoint["model_state_dict"].keys())

    print("MODEL PARAMS ======= ",checkpoint["params"])
    print("META PARAMS ======= ",checkpoint["meta_params"])
    
    # Get model parameters
    model_params = checkpoint.get('params', {})
    print(f"\nModel type: {model_params.get('model')}")
    print(f"Filter settings: sid_filtered={model_params.get('sid_filtered')}, meta_filtered={model_params.get('meta_filtered')}")
    
    # Create model
    model = MAMLModel(
        n_question=model_params.get('n_question', 948),        
        question_dim=1,  # biirt
        dropout=model_params.get('dropout', 0.2),
        sampling=model_params.get('sampling', 'active'),
        n_query=model_params.get('n_query', 10),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Inspect question difficulties
    question_diff = model.question_difficulty.data.cpu().numpy()
    print(f"\nQuestion difficulty parameter shape: {question_diff.shape}")
    print(f"Question difficulty stats:")
    print(f"  Mean: {question_diff.mean():.4f}")
    print(f"  Std: {question_diff.std():.4f}")
    print(f"  Min: {question_diff.min():.4f}")
    print(f"  Max: {question_diff.max():.4f}")
    
    # Check for zero or near-zero difficulties
    print(f"\n{'='*60}")
    print("ZERO DIFFICULTY ANALYSIS")
    print(f"{'='*60}")
    
    # Exact zeros
    zero_mask = (question_diff == 0.0)
    n_zeros = zero_mask.sum()
    print(f"\nExact zeros: {n_zeros} questions")
    if n_zeros > 0:
        zero_indices = np.where(zero_mask.flatten())[0]
        print(f"Question IDs with exact zero difficulty: {zero_indices.tolist()}")
    
    # Near-zero (within different thresholds)
    for threshold in [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1]:
        near_zero_mask = (np.abs(question_diff) < threshold)
        n_near_zeros = near_zero_mask.sum()
        print(f"\nWithin {threshold}: {n_near_zeros} questions")
        if n_near_zeros > 0 and n_near_zeros <= 50:  # Only show IDs if not too many
            near_zero_indices = np.where(near_zero_mask.flatten())[0]
            print(f"  Question IDs: {near_zero_indices.tolist()}")
            # Show their actual values
            if n_near_zeros <= 20:
                for idx in near_zero_indices:
                    print(f"    Q{idx}: {question_diff[0, idx]:.8f}")
    
    print(f"\n{'='*60}")
    
    # Show first 20 question difficulties
    print(f"\nFirst 20 question difficulties:")
    for i in range(20):
        print(f"  Q{i}: {question_diff[0, i]:.4f}")
    
    # Show some high-numbered questions (like 740, 716, 852)
    print(f"\nSome high-numbered question difficulties (740, 716, 852, 885):")
    for q_id in [740, 716, 852, 885]:
        print(f"  Q{q_id}: {question_diff[0, q_id]:.4f}")
    
    # Check variance across questions
    print(f"\nStd dev across all questions: {question_diff.std():.6f}")
    
    # Count how many questions have very similar difficulties (within 0.1 of each other)
    diff_range = question_diff.max() - question_diff.min()
    print(f"\nRange of difficulties: {diff_range:.4f}")
    
    # Histogram of difficulties
    print(f"\nHistogram of difficulties:")
    hist, bins = np.histogram(question_diff, bins=10)
    for i, (count, bin_start) in enumerate(zip(hist, bins[:-1])):
        print(f"  [{bin_start:.2f}, {bins[i+1]:.2f}): {count} questions")
    
    return model, question_diff, checkpoint

if __name__ == "__main__":
    # Load both models
    filtered_model, filtered_diff, filtered_checkpoint = load_and_inspect_model(
        "saved_models_exp1/actives/bobcat_final_155_True_eedi-3_biirt-active_10_20251109_134224.pt",
        "FILTERED (meta_filtered=True)"
    )
    
    unfiltered_model, unfiltered_diff, unfiltered_checkpoint = load_and_inspect_model(
        "saved_models_exp1/actives/bobcat_final_155_False_eedi-3_biirt-active_10_20251115_182957.pt",
        "UNFILTERED (meta_filtered=False)"
    )
    

    # print(filtered_model["meta_params"])
    # print(unfiltered_model["meta_params"])
    
    # Compare the two
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    
    print("\nDifference in question difficulties (unfiltered - filtered):")
    diff_comparison = unfiltered_diff - filtered_diff
    print(f"  Mean difference: {diff_comparison.mean():.4f}")
    print(f"  Std of difference: {diff_comparison.std():.4f}")
    print(f"  Max absolute difference: {np.abs(diff_comparison).max():.4f}")
    
    print("\nQuestions with biggest differences:")
    top_diffs = np.argsort(np.abs(diff_comparison.flatten()))[-10:][::-1]
    for q_id in top_diffs:
        print(f"  Q{q_id}: filtered={filtered_diff[0, q_id]:.4f}, unfiltered={unfiltered_diff[0, q_id]:.4f}, diff={diff_comparison[0, q_id]:.4f}")
    
    print("\nFirst 20 questions comparison:")
    for i in range(20):
        print(f"  Q{i}: filtered={filtered_diff[0, i]:.4f}, unfiltered={unfiltered_diff[0, i]:.4f}")
    
    # Compare zeros between models
    print(f"\n{'='*80}")
    print("ZERO DIFFICULTY COMPARISON")
    print(f"{'='*80}")
    
    filtered_zeros = (filtered_diff == 0.0)
    unfiltered_zeros = (unfiltered_diff == 0.0)
    
    print(f"\nFiltered model: {filtered_zeros.sum()} exact zeros")
    print(f"Unfiltered model: {unfiltered_zeros.sum()} exact zeros")
    
    # Questions that are zero in one but not the other
    only_filtered_zero = filtered_zeros & ~unfiltered_zeros
    only_unfiltered_zero = unfiltered_zeros & ~filtered_zeros
    both_zero = filtered_zeros & unfiltered_zeros
    
    print(f"\nZero only in filtered: {only_filtered_zero.sum()}")
    if only_filtered_zero.sum() > 0:
        indices = np.where(only_filtered_zero.flatten())[0]
        print(f"  Question IDs: {indices.tolist()}")
    
    print(f"\nZero only in unfiltered: {only_unfiltered_zero.sum()}")
    if only_unfiltered_zero.sum() > 0:
        indices = np.where(only_unfiltered_zero.flatten())[0]
        print(f"  Question IDs: {indices.tolist()}")
    
    print(f"\nZero in both models: {both_zero.sum()}")
    if both_zero.sum() > 0:
        indices = np.where(both_zero.flatten())[0]
        print(f"  Question IDs: {indices.tolist()}")
    
    # Non-zero questions in filtered model
    non_zero_filtered = ~filtered_zeros
    non_zero_indices = np.where(non_zero_filtered.flatten())[0]
    print(f"\nNon-zero questions in filtered model: {non_zero_indices.tolist()}")

    