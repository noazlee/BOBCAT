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

    print("\nMODEL PARAMS:", checkpoint["params"])
    print("\nMETA PARAMS:", checkpoint["meta_params"])
    
    # Get model parameters
    model_params = checkpoint.get('params', {})
    print(f"\nModel type: {model_params.get('model')}")
    print(f"Filter settings: sid_filtered={model_params.get('sid_filtered')}, meta_filtered={model_params.get('meta_filtered')}")
    
    # Create model
    model = MAMLModel(
        n_question=model_params.get('n_question', 948),        
        question_dim=model_params.get('question_dim', 4),
        dropout=model_params.get('dropout', 0.2),
        sampling=model_params.get('sampling', 'biased'),
        n_query=model_params.get('n_query', 10),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # CORRECT UNDERSTANDING: This model architecture is:
    # student_embed [batch, 4] -> Linear(4, 256) -> ReLU -> Dropout -> Linear(256, 948) -> predictions [batch, 948]
    
    print(f"\n{'='*60}")
    print("ARCHITECTURE ANALYSIS")
    print(f"{'='*60}")
    print("\nThis BINN model works as follows:")
    print("  1. Takes a 4D student representation (meta-parameter)")
    print("  2. Passes through: Linear(4->256) -> ReLU -> Dropout")
    print("  3. Output layer: Linear(256->948) predicts ALL questions at once")
    print("  4. Does NOT have separate embeddings per question")
    
    print(f"\nModel parameters:")
    total_params = 0
    for name, param in model.named_parameters():
        n_params = param.numel()
        total_params += n_params
        print(f"  {name}: shape {param.shape}, {n_params:,} parameters")
    print(f"\nTotal parameters: {total_params:,}")
    
    # Get the actual learned weights
    first_layer_weight = model.layers[0].weight.data.cpu().numpy()  # [256, 4]
    first_layer_bias = model.layers[0].bias.data.cpu().numpy()      # [256]
    output_layer_weight = model.output_layer.weight.data.cpu().numpy()  # [948, 256]
    output_layer_bias = model.output_layer.bias.data.cpu().numpy()      # [948]
    
    print(f"\n{'='*60}")
    print("LAYER WEIGHT STATISTICS")
    print(f"{'='*60}")
    
    print(f"\nFirst layer (input->hidden):")
    print(f"  Shape: {first_layer_weight.shape}")
    print(f"  Mean: {first_layer_weight.mean():.4f}")
    print(f"  Std: {first_layer_weight.std():.4f}")
    print(f"  Min: {first_layer_weight.min():.4f}")
    print(f"  Max: {first_layer_weight.max():.4f}")
    print(f"  Non-zero elements: {(first_layer_weight != 0).sum()} / {first_layer_weight.size}")
    
    print(f"\nOutput layer (hidden->questions):")
    print(f"  Shape: {output_layer_weight.shape}")
    print(f"  Mean: {output_layer_weight.mean():.4f}")
    print(f"  Std: {output_layer_weight.std():.4f}")
    print(f"  Min: {output_layer_weight.min():.4f}")
    print(f"  Max: {output_layer_weight.max():.4f}")
    print(f"  Non-zero elements: {(output_layer_weight != 0).sum()} / {output_layer_weight.size}")
    
    # Analyze per-question output weights
    print(f"\n{'='*60}")
    print("PER-QUESTION OUTPUT ANALYSIS")
    print(f"{'='*60}")
    
    # For each question, look at its output weights (row in output layer)
    question_output_norms = np.linalg.norm(output_layer_weight, axis=1)  # L2 norm of each row
    question_output_means = np.abs(output_layer_weight).mean(axis=1)     # Mean absolute weight
    
    print(f"\nPer-question output weight norms (how much each question's prediction depends on hidden layer):")
    print(f"  Mean norm: {question_output_norms.mean():.4f}")
    print(f"  Std norm: {question_output_norms.std():.4f}")
    print(f"  Min norm: {question_output_norms.min():.4f}")
    print(f"  Max norm: {question_output_norms.max():.4f}")
    
    # Check for questions with very low weights
    low_threshold = 0.1
    low_weight_questions = question_output_norms < low_threshold
    n_low = low_weight_questions.sum()
    print(f"\nQuestions with output norm < {low_threshold}: {n_low}")
    
    # Show first 20 questions
    print(f"\nFirst 20 questions output norms:")
    for i in range(20):
        print(f"  Q{i}: norm={question_output_norms[i]:.4f}, mean_abs_weight={question_output_means[i]:.4f}, bias={output_layer_bias[i]:.4f}")
    
    # Show some of the sampled questions from your CSV
    print(f"\nSampled questions from CSV (882, 642, 291, 24, 668):")
    for q_id in [882, 642, 291, 24, 668]:
        if q_id < len(question_output_norms):
            print(f"  Q{q_id}: norm={question_output_norms[q_id]:.4f}, mean_abs_weight={question_output_means[q_id]:.4f}, bias={output_layer_bias[q_id]:.4f}")
    
    # Histogram
    print(f"\nHistogram of output weight norms:")
    hist, bins = np.histogram(question_output_norms, bins=10)
    for i, (count, bin_start) in enumerate(zip(hist, bins[:-1])):
        print(f"  [{bin_start:.2f}, {bins[i+1]:.2f}): {count} questions")
    
    # Test the model with a sample input
    print(f"\n{'='*60}")
    print("MODEL OUTPUT TEST")
    print(f"{'='*60}")
    
    model.eval()
    with torch.no_grad():
        # Use the meta parameters from checkpoint
        meta_param = checkpoint['meta_params'][0]  # [1, 4]
        print(f"\nMeta parameter: {meta_param}")
        
        # Pass through model
        hidden = model.layers(meta_param)  # [1, 256]
        output = model.output_layer(hidden)  # [1, 948]
        output_probs = torch.sigmoid(output)  # [1, 948]
        
        print(f"\nModel output statistics:")
        print(f"  Output logits - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
        print(f"  Output probs - Mean: {output_probs.mean():.4f}, Std: {output_probs.std():.4f}")
        print(f"  Output probs - Min: {output_probs.min():.4f}, Max: {output_probs.max():.4f}")
        
        # Show predictions for first 20 questions
        print(f"\nPredictions for first 20 questions:")
        for i in range(20):
            print(f"  Q{i}: logit={output[0, i]:.4f}, prob={output_probs[0, i]:.4f}")
        
        # Show predictions for sampled questions
        print(f"\nPredictions for sampled questions (882, 642, 291, 24, 668):")
        for q_id in [882, 642, 291, 24, 668]:
            if q_id < output.shape[1]:
                print(f"  Q{q_id}: logit={output[0, q_id]:.4f}, prob={output_probs[0, q_id]:.4f}")
    
    analysis = {
        'first_layer_weight': first_layer_weight,
        'first_layer_bias': first_layer_bias,
        'output_layer_weight': output_layer_weight,
        'output_layer_bias': output_layer_bias,
        'question_output_norms': question_output_norms,
        'question_output_means': question_output_means,
    }
    
    return model, analysis, checkpoint

if __name__ == "__main__":
    print("\n" + "="*80)
    print("LOADING AND COMPARING BINN MODELS")
    print("="*80)
    
    filtered_model, filtered_analysis, filtered_checkpoint = load_and_inspect_model(
        "saved_models_exp1/bobcat_final_155_True_eedi-3_binn-biased_10_20251119_013006.pt",
        "FILTERED (meta_filtered=True)"
    )
    
    unfiltered_model, unfiltered_analysis, unfiltered_checkpoint = load_and_inspect_model(
        "saved_models_exp1/bobcat_final_155_False_eedi-3_binn-biased_10_20251119_005906.pt",
        "UNFILTERED (meta_filtered=False)"
    )
    
    # Compare the two models
    print(f"\n{'='*80}")
    print("DETAILED COMPARISON")
    print(f"{'='*80}")
    
    # Compare output layer weights
    print(f"\n--- Output Layer Weight Comparison ---")
    output_diff = unfiltered_analysis['output_layer_weight'] - filtered_analysis['output_layer_weight']
    print(f"\nDifference in output layer weights (unfiltered - filtered):")
    print(f"  Mean difference: {output_diff.mean():.6f}")
    print(f"  Std of difference: {output_diff.std():.6f}")
    print(f"  Max absolute difference: {np.abs(output_diff).max():.6f}")
    
    # Per-question output norm comparison
    filtered_norms = filtered_analysis['question_output_norms']
    unfiltered_norms = unfiltered_analysis['question_output_norms']
    norm_diff = unfiltered_norms - filtered_norms
    
    print(f"\n--- Per-Question Output Norm Comparison ---")
    print(f"Mean norm difference: {norm_diff.mean():.6f}")
    print(f"Std norm difference: {norm_diff.std():.6f}")
    print(f"Max absolute norm difference: {np.abs(norm_diff).max():.6f}")
    
    print(f"\nQuestions with biggest norm differences:")
    top_diff_indices = np.argsort(np.abs(norm_diff))[-20:][::-1]
    for i, q_id in enumerate(top_diff_indices):
        print(f"  Q{q_id}: filtered={filtered_norms[q_id]:.4f}, unfiltered={unfiltered_norms[q_id]:.4f}, diff={norm_diff[q_id]:.4f}")
    
    # Correlation
    correlation = np.corrcoef(filtered_norms, unfiltered_norms)[0, 1]
    print(f"\nCorrelation between filtered and unfiltered output norms: {correlation:.4f}")
    
    # Compare first layer
    print(f"\n--- First Layer Comparison ---")
    first_layer_diff = unfiltered_analysis['first_layer_weight'] - filtered_analysis['first_layer_weight']
    print(f"Difference in first layer weights:")
    print(f"  Mean difference: {first_layer_diff.mean():.6f}")
    print(f"  Std of difference: {first_layer_diff.std():.6f}")
    print(f"  Max absolute difference: {np.abs(first_layer_diff).max():.6f}")
    
    # Compare meta parameters
    print(f"\n--- Meta Parameter Comparison ---")
    filtered_meta = filtered_checkpoint['meta_params'][0].detach().cpu().numpy()
    unfiltered_meta = unfiltered_checkpoint['meta_params'][0].detach().cpu().numpy()
    print(f"Filtered meta params: {filtered_meta}")
    print(f"Unfiltered meta params: {unfiltered_meta}")
    print(f"Difference: {unfiltered_meta - filtered_meta}")
    print(f"L2 norm of difference: {np.linalg.norm(unfiltered_meta - filtered_meta):.6f}")
    print(f"  - Correlation in output norms: {correlation:.4f}")
    print(f"  - Max norm difference: {np.abs(norm_diff).max():.4f}")
    print(f"  - This suggests {'SIMILAR' if correlation > 0.9 else 'DIFFERENT'} learned representations")