import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import csv
from datetime import datetime

from model import MAMLModel, device
from dataset2 import Dataset, collate_fn
from utils.utils import open_json, dump_json, compute_auc, compute_accuracy, data_split, try_makedirs

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("saved_models_exp1/bobcat_final_163_164_165_False_eedi-3_biirt-active_10_20251112_224101.pt", map_location=device)
    print(checkpoint.keys())
    print(checkpoint["model_state_dict"].keys())
    print(checkpoint["params"].keys())
    print(checkpoint["params"]["model"])
    
    # First, check the state dict to determine question_dim
    state_dict = checkpoint['model_state_dict']
    # Get model parameters
    model_params = checkpoint.get('params', {})
    
    # Create model with correct architecture
    model = MAMLModel(
        n_question=model_params.get('n_question'),        
        question_dim=1, # biirt
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
    
    return model, meta_params, model_params

def run_random_test(batch, model, n_query=10, question_dim=1):
    
    # Initialize student parameters
    batch_size = len(batch['input_labels'])
    student_params = torch.zeros(batch_size, question_dim).to(device)
    student_params.requires_grad = True
    
    config = {
        'available_mask': batch['input_mask'].to(device).clone(),   # what can be sampled (512, 948)
        'train_mask': torch.zeros(batch_size, model.n_question).long().to(device),  # what has been sampled (512, 948)
        'mode': 'test',
        'meta_param': student_params
    }
    
    # Track sampled questions for each student
    sampled_questions = []
    
    # Active sampling loop
    for query_idx in range(n_query):
        # Pick uncertain sample
        with torch.no_grad():
            output = model.compute_output(student_params)
            output_probs = torch.sigmoid(output)
            
            # Calculate uncertainty scores
            inf_mask = torch.clamp(
                torch.log(config['available_mask'].float()), 
                min=torch.finfo(torch.float32).min
            )
            scores = torch.min(1-output_probs, output_probs) + inf_mask
            actions = torch.argmax(scores, dim=-1)
        
        # Update masks and track questions
        for i in range(batch_size):
            action = actions[i].item()
            config['train_mask'][i, action] = 1
            config['available_mask'][i, action] = 0
            sampled_questions.append(action)
        
        # Inner loop update (gradient descent on student params)
        res = model(batch, config)
        train_loss = res['train_loss']
        
        # Compute gradients and update
        grad = torch.autograd.grad(train_loss, student_params, retain_graph=True)[0]
        student_params = student_params - 0.001 * grad  # learning rate = 0.001
        student_params = student_params.detach().requires_grad_(True)
        config['meta_param'] = student_params
    
    # Final prediction
    with torch.no_grad():
        res = model(batch, config) 
    
    # Reshape sampled questions by student
    sampled_questions = np.array(sampled_questions).reshape(n_query, batch_size).T 
    
    return res['output'], sampled_questions # pred probs for each q for each student, questions chosen

def test_model(model, test_data, n_query=10, question_dim=1):
    
    # Create dataset and dataloader
    test_dataset = Dataset(test_data) # this splits training/meta set
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
        predictions, sampled_q_indices = run_random_test(batch, model, n_query, question_dim)
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
        fieldnames = ['user_id', 'sampled_q_ids', 'sampled_subject_ids', 'student_total_questions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in sampled_data:
            writer.writerow({
                'user_id': data['user_id'],
                'sampled_q_ids': ','.join(map(str, data['sampled_q_ids'])),
                'sampled_subject_ids': ','.join(map(str, data['sampled_subject_ids'])),
                'student_total_questions': data['student_total_questions']
            })

if __name__ == "__main__":
    
    # Load model
    print("Loading model...")
    model, meta_params, model_params = load_model()
    print(f"Model loaded with {model_params.get('n_question', 948)} questions")
    
    # Load test data
    print("Loading test data...")
    train_data, valid_data, test_data = data_split('data/train_task_eedi-3.json', fold=1, seed=221)
    print(f"Test set size: {len(test_data)} students")
    
    # Run testing
    print("Running test...")
    auc, accuracy, sampled_data = test_model(
        model, 
        test_data, 
        n_query=model_params.get('n_query', 10),
        question_dim=1
    )
    
    # Print results
    print(f"=== Test Results ===")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save sampled questions to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'outputs/sampled_questions_eedi3_active_{timestamp}.csv'
    save_questions_to_csv(sampled_data, csv_filename)
    print(f"Sampled questions saved to {csv_filename}")
    
    print("Sample of sampled questions (first 3 students):")
    for i, data in enumerate(sampled_data[:3]):
        print(f"\nStudent {data['user_id']}:")
        print(f"  Sampled Q IDs: {data['sampled_q_ids'][:5]}... (showing first 5)")
        print(f"  Sampled Subjects: {data['sampled_subject_ids']}")