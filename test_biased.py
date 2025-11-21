import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import csv

from model import MAMLModel, device
from dataset2 import Dataset, collate_fn
from utils.utils import open_json, dump_json, compute_auc, compute_accuracy, data_split, try_makedirs

from datetime import datetime
from policy import StraightThrough

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("saved_models_biased_real/bobcat_final_eedi-3_binn-biased_10_20251120_155725.pt", map_location=device)
    print(checkpoint.keys())
    print(checkpoint["model_state_dict"].keys())
    print(checkpoint["params"].keys())
    print(checkpoint["params"]["model"])

   
    # First, check the state dict to determine question_dim
    state_dict = checkpoint['model_state_dict']
    # Get model parameters
    model_params = checkpoint.get('params', {})

    print(f"Checkpoint test_auc: {checkpoint['test_auc']}")
    print(f"Params inner_loop: {model_params['inner_loop']}")
    print(f"Params repeat: {model_params['repeat']}")
    
    betas = (0.9, 0.999)
    st_policy = StraightThrough(
        model_params.get("n_question", 948), 
        model_params.get("n_question", 948),
        model_params.get("policy_lr", 2e-3), 
        betas
    )

    if 'st_policy_state_dict' in checkpoint:
        st_policy.policy.load_state_dict(checkpoint['st_policy_state_dict'])
        print("Loaded trained policy weights")
    else:
        print("WARNING: No policy weights found in checkpoint!")

    # Create model with correct architecture
    model = MAMLModel(
        n_question=model_params.get('n_question'),        
        question_dim=4, # binn
        dropout=model_params.get('dropout', 0.2),
        sampling=model_params.get('sampling', 'active'),
        n_query=model_params.get('n_query', 10),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load meta params if available
    meta_params = None
    if "meta_params" in checkpoint:
        meta_params = checkpoint["meta_params"][0]
        meta_params = meta_params.to(device)

    inner_loop = model_params['inner_loop']
    inner_lr = model_params['inner_lr']
    
    return model, meta_params, model_params, st_policy, inner_loop, inner_lr


def pick_biased_samples(st_policy, batch, model, config, n_query, student_params):
    """
    Pick samples using the biased policy network.
    Similar to the training loop's pick_biased_samples function.
    """
    batch_size = len(batch['input_labels'])
    
    env_states = model.reset(batch)
    action_mask = env_states['action_mask']
    train_mask = env_states['train_mask']
    
    sampled_questions = []

    for query_idx in range(n_query):
        with torch.no_grad():
            state = model.step(env_states)
            
        # Get actions from policy
        with torch.no_grad():
            train_mask_sample, actions = st_policy.policy(state, action_mask)

        # Track sampled questions (actions are the question indices)
        sampled_questions.append(actions.cpu().numpy())
        
        # Update masks
        action_mask[range(len(action_mask)), actions] = 0
        env_states['train_mask'] = train_mask + train_mask_sample.data
        env_states['action_mask'] = action_mask
        train_mask = env_states['train_mask']
        
        # Inner loop update with current student params
        config['train_mask'] = train_mask
        config['meta_param'] = student_params

        for _ in range(inner_loop):
            res = model(batch, config)
            train_loss = res['train_loss']
            
            # Compute gradients and update
            grad = torch.autograd.grad(train_loss, student_params, retain_graph=True)[0]
            student_params = student_params - inner_lr * grad
            student_params = student_params.detach().requires_grad_(True)
            config['meta_param'] = student_params 
        
    
    # Set final train mask and params
    config['train_mask'] = env_states['train_mask']
    config['meta_param'] = student_params
    
    # Convert to (batch_size, n_query) format
    sampled_q_indices = np.array(sampled_questions).T  # (batch_size, n_query)
    
    return config, sampled_q_indices

def run_biased_test(st_policy, batch, model, n_query=10, question_dim=4):
    """
    Run test with biased policy sampling.
    """
    batch_size = len(batch['input_labels'])
    
    # Initialize student parameters (same as active sampling in test.py)
    student_params = meta_params.expand(batch_size, -1).clone().detach()
    student_params.requires_grad = True
    
    # Initialize configuration
    config = {
        'available_mask': batch['input_mask'].to(device).clone(),
        'train_mask': torch.zeros(batch_size, model.n_question).long().to(device),
        'mode': 'test',
        'question_dim': question_dim,
        'meta_param': student_params
    }
    
    # Pick samples using biased policy
    config, sampled_questions = pick_biased_samples(st_policy, batch, model, config, n_query, student_params)
    
    # Final prediction with sampled questions
    with torch.no_grad():
        res = model(batch, config)
    
    return res['output'], sampled_questions

def test_model_multiple_seeds(st_policy, model, test_data, n_query=10, question_dim=4, 
                               subject_filter=[163,164,165], num_seeds=10):
    all_aucs = []
    all_accs = []
    all_sampled_data = []
    
    for seed_idx, seed in enumerate(range(100, 100 + num_seeds)):
        print(f"\n[{seed_idx+1}/{num_seeds}] Testing with seed {seed}...")
        
        # Create fresh dataset with this seed
        if subject_filter is not None:
            test_dataset = Dataset(test_data, subject_filter, filter_by_meta_set=True)
        else:
            test_dataset = Dataset(test_data)
        
        test_dataset.seed = seed  # Set seed like in training
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=collate_fn(model.n_question),
            num_workers=0
        )
        
        # Run test for this seed
        predictions_list = []
        targets_list = []
        sampled_data = []
        
        for batch_idx, batch in enumerate(test_loader):
            predictions, sampled_q_indices = run_biased_test(st_policy, batch, model, n_query, question_dim)
            
            targets = batch['output_labels'].float().numpy()
            masks = batch['output_mask'].numpy() == 1
            
            for i in range(len(predictions)):
                mask = masks[i]
                predictions_list.extend(predictions[i][mask])
                targets_list.extend(targets[i][mask])
                
                # Store sampled questions
                user_id = batch['user_ids'][i]
                all_q_ids = batch['all_q_ids'][i]
                all_subject_ids = batch['all_subject_ids'][i]
                sampled_q_ids = sampled_q_indices[i].tolist()
                
                q_to_subject = {}
                for j, q_id in enumerate(all_q_ids):
                    if j < len(all_subject_ids):
                        q_to_subject[q_id] = all_subject_ids[j] if isinstance(all_subject_ids[j], list) else [all_subject_ids[j]]
                
                sampled_subjects = []
                for q_id in sampled_q_ids:
                    if q_id in q_to_subject:
                        sampled_subjects.extend(q_to_subject[q_id])
                
                sampled_data.append({
                    'user_id': user_id,
                    'seed': seed,
                    'sampled_q_ids': sampled_q_ids,
                    'sampled_subject_ids': list(sampled_subjects),
                    'student_total_questions': len(all_q_ids)
                })
        
        # Calculate metrics for this seed
        predictions_array = np.array(predictions_list)
        targets_array = np.array(targets_list)
        
        auc = compute_auc(targets_array, predictions_array)
        accuracy = compute_accuracy(targets_array, predictions_array)
        
        all_aucs.append(auc)
        all_accs.append(accuracy)
        all_sampled_data.extend(sampled_data)
        
        print(f"  Seed {seed}: AUC={auc:.4f}, Acc={accuracy:.4f}")
    
    # Average results
    avg_auc = np.mean(all_aucs)
    avg_acc = np.mean(all_accs)
    
    print(f"\nAUC std dev: {np.std(all_aucs):.4f}")
    print(f"Accuracy std dev: {np.std(all_accs):.4f}")
    
    return avg_auc, avg_acc, all_sampled_data

def test_model(st_policy, model, test_data, n_query=10, question_dim=4, subject_filter=[163,164,165]):
    """
    Test model with biased sampling policy.
    
    Args:
        st_policy: StraightThrough policy for biased sampling
        model: The trained model
        test_data: List of test student data
        n_query: Number of questions to sample
        question_dim: Dimension of question embeddings
        subject_filter: Subject ID to filter by (e.g., 155), or None for no filtering
    """
    
    # Create dataset and dataloader
    if subject_filter is not None:
        test_dataset = Dataset(test_data, subject_filter, filter_by_meta_set=True)
    else:
        test_dataset = Dataset(test_data)
        
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn(model.n_question),
        num_workers=0
    )
    
    all_predictions = []
    all_targets = []
    sampled_data = []  # Will store user_id, sampled q_ids, subject_ids
    
    print(f"Testing on {len(test_dataset)} students...")
    
    for batch_idx, batch in enumerate(test_loader):
        # batch: user_ids, input_labels, input_mask, output_labels, output_mask, all_q_ids, all_subject_ids
        # Run test with sampling
        predictions, sampled_q_indices = run_biased_test(st_policy, batch, model, n_query, question_dim)
        # 32x948,       32x10

        # Get targets and masks
        targets = batch['output_labels'].float().numpy()
        masks = batch['output_mask'].numpy() == 1
        
        # Store predictions
        for i in range(len(predictions)):
            mask = masks[i]
            all_predictions.extend(predictions[i][mask])
            all_targets.extend(targets[i][mask])
            
            # Store sampled questions data
            user_id = batch['user_ids'][i]
            all_q_ids = batch['all_q_ids'][i]
            all_subject_ids = batch['all_subject_ids'][i]
            
            # The sampled indices ARE the question IDs (0-947)
            sampled_q_ids = sampled_q_indices[i].tolist()
            
            # Debug for first student
            if batch_idx == 0 and i == 0:
                print(f"\n=== DEBUG: First student question mapping ===")
                print(f"all_q_ids type: {type(all_q_ids)}")
                print(f"all_q_ids length: {len(all_q_ids)}")
                print(f"all_q_ids first 10: {all_q_ids[:10]}")
                print(f"sampled action indices: {sampled_q_indices[i].tolist()}")
                print(f"input_mask shape: {batch['input_mask'][i].shape}")
                print(f"Questions where input_mask==1: {torch.where(batch['input_mask'][i]==1)[0][:10].tolist()}")
            
            # Get the subjects for the sampled questions
            # First create a mapping of q_id to subject_ids from the student's data
            q_to_subject = {}
            for j, q_id in enumerate(all_q_ids):
                if j < len(all_subject_ids):
                    q_to_subject[q_id] = all_subject_ids[j] if isinstance(all_subject_ids[j], list) else [all_subject_ids[j]]
            
            # Now get subjects for sampled questions
            sampled_subjects = []
            for q_id in sampled_q_ids:
                if q_id in q_to_subject:
                    sampled_subjects.extend(q_to_subject[q_id])
            
            sampled_data.append({
                'user_id': user_id,
                'sampled_q_ids': sampled_q_ids,  # These ARE the actual question IDs (0-947)
                'sampled_subject_ids': list(sampled_subjects),
                'student_total_questions': len(all_q_ids)
            })
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {(batch_idx + 1) * 32} students...")
    
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    auc = compute_auc(all_targets, all_predictions)
    accuracy = compute_accuracy(all_targets, all_predictions)
    
    return auc, accuracy, sampled_data

def save_questions_to_csv(sampled_data, filename='sampled_questions.csv'):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['user_id', 'seed', 'sampled_q_ids', 'sampled_subject_ids', 'student_total_questions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in sampled_data:
            writer.writerow({
                'user_id': data['user_id'],
                'seed': data.get('seed', 'N/A'),
                'sampled_q_ids': ','.join(map(str, data['sampled_q_ids'])),
                'sampled_subject_ids': ','.join(map(str, data['sampled_subject_ids'])),
                'student_total_questions': data['student_total_questions']
            })

if __name__ == "__main__":
    
    # Load model
    print("Loading model...")
    model, meta_params, model_params, st_policy, inner_loop, inner_lr = load_model()
    print(f"Model loaded with {model_params.get('n_question', 948)} questions")
    
    # Load test data
    print("Loading test data...")
    train_data, valid_data, test_data = data_split('data/train_task_eedi-3.json', fold=1, seed=221)
    print(f"Test set size: {len(test_data)} students")
    
    # Initialize policy

    # Determine if filtering by subject
    subject_filter = model_params.get('sid_filtered', None)
    print(f"Subject filter: {subject_filter}")

    # Run testing
    print("Running test with biased policy...")
    avg_auc, avg_accuracy, all_sampled_data = test_model_multiple_seeds(
        st_policy,
        model, 
        test_data, 
        n_query=model_params.get('n_query', 10),
        question_dim=4,
        subject_filter=[163,164,165],
        num_seeds=10
    )
    
    # Print results
    print("\n" + "="*60)
    print(f"=== AVERAGED Test Results (over 10 seeds) ===")
    print("="*60)
    print(f"Average AUC: {avg_auc:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    
    # Save sampled questions to CSV (from all seeds)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filter_str = f"_{subject_filter}" if subject_filter else ""
    csv_filename = f'outputs/sampled_questions_eedi3_biased{filter_str}_multiSeed_{timestamp}.csv'
    save_questions_to_csv(all_sampled_data, csv_filename)
    print(f"\nAll sampled questions saved to {csv_filename}")
    print(f"Total students across all seeds: {len(all_sampled_data)}")