import numpy as np
import torch
import os
from dataset2 import Dataset, collate_fn
from utils.utils import data_split
from collections import defaultdict, Counter
import json
from datetime import datetime

# Open output file
output_file = open(f'investigation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w')

def log(msg=""):
    print(msg)
    output_file.write(msg + "\n")
    output_file.flush()

def investigate_dataset_filtering():
    # Investigate what questions are available in filtered vs unfiltered datasets
    
    log("="*80)
    log("INVESTIGATING DATASET FILTERING")
    log("="*80)
    
    # Load data
    data_path = 'data/train_task_eedi-3.json'
    train_data, valid_data, test_data = data_split(data_path, fold=1, seed=221)
    
    log(f"\nOriginal data sizes:")
    log(f"  Train: {len(train_data)} students")
    log(f"  Valid: {len(valid_data)} students")
    log(f"  Test: {len(test_data)} students")
    
    # Analyze a few students before filtering
    log("\n" + "="*80)
    log("ANALYZING STUDENTS BEFORE FILTERING")
    log("="*80)
    
    for i in range(min(3, len(train_data))):
        student = train_data[i]
        total_qs = len(student['q_ids'])
        
        # Count questions with subject 155
        subject_155_count = 0
        subject_155_qids = []
        all_subjects = set()
        
        for j, subject_ids in enumerate(student['subject_ids']):
            all_subjects.update(subject_ids)
            if 155 in subject_ids:
                subject_155_count += 1
                subject_155_qids.append(student['q_ids'][j])
        
        log(f"\nStudent {i} (user_id: {student['user_id']}):")
        log(f"  Total questions: {total_qs}")
        log(f"  Questions with subject 155: {subject_155_count} ({100*subject_155_count/total_qs:.1f}%)")
        log(f"  Total unique subjects: {len(all_subjects)}")
        log(f"  Subject 155 q_ids (first 10): {subject_155_qids[:10]}")
    
    # Create filtered datasets
    log("\n" + "="*80)
    log("CREATING FILTERED DATASETS")
    log("="*80)
    
    # Dataset with meta_filtered=True (what the filtered model uses)
    train_dataset_filtered = Dataset(train_data, filter_id=155, filter_by_meta_set=True)
    
    # Dataset with meta_filtered=False (what the unfiltered model uses)
    train_dataset_unfiltered = Dataset(train_data, filter_id=155, filter_by_meta_set=False)
    
    log(f"\nFiltered dataset size (meta_filtered=True): {len(train_dataset_filtered)} students")
    log(f"Unfiltered dataset size (meta_filtered=False): {len(train_dataset_unfiltered)} students")
    
    return train_dataset_filtered, train_dataset_unfiltered, train_data


def analyze_batches(dataset, dataset_name, n_batches=5):
    # Analyze what questions appear in training batches
    
    log("\n" + "="*80)
    log(f"ANALYZING BATCHES: {dataset_name}")
    log("="*80)
    
    collate_fn_obj = collate_fn(n_question=948)
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        collate_fn=collate_fn_obj, 
        batch_size=32, 
        num_workers=0, 
        shuffle=False,  # Don't shuffle so we can track specific students
        drop_last=False
    )
    
    all_input_questions = set()
    all_output_questions = set()
    all_questions = set()
    
    student_question_counts = []
    student_input_counts = []
    student_output_counts = []
    
    question_frequency = defaultdict(int)
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= n_batches:
            break
        
        log(f"\n--- Batch {batch_idx} ---")
        log(f"Batch size: {len(batch['user_ids'])} students")
        
        # Analyze first 3 students in batch
        for i in range(min(3, len(batch['user_ids']))):
            user_id = batch['user_ids'][i]
            
            # Get input (trainable) questions
            input_mask = batch['input_mask'][i]
            input_q_indices = torch.where(input_mask == 1)[0].cpu().numpy()
            
            # Get output (meta) questions
            output_mask = batch['output_mask'][i]
            output_q_indices = torch.where(output_mask == 1)[0].cpu().numpy()
            
            # Track all questions
            all_input_questions.update(input_q_indices)
            all_output_questions.update(output_q_indices)
            all_questions.update(input_q_indices)
            all_questions.update(output_q_indices)
            
            student_question_counts.append(len(input_q_indices) + len(output_q_indices))
            student_input_counts.append(len(input_q_indices))
            student_output_counts.append(len(output_q_indices))
            
            # Count frequency
            for q in input_q_indices:
                question_frequency[q] += 1
            for q in output_q_indices:
                question_frequency[q] += 1
            
            log(f"\n  Student {i} (user_id: {user_id}):")
            log(f"    Input (trainable) questions: {len(input_q_indices)}")
            log(f"    Output (meta) questions: {len(output_q_indices)}")
            log(f"    Total questions: {len(input_q_indices) + len(output_q_indices)}")
            log(f"    Input q_ids (first 20): {input_q_indices[:20].tolist()}")
            log(f"    Output q_ids (first 20): {output_q_indices[:20].tolist()}")
            
            # Check subject 155 in output
            all_q_ids = batch['all_q_ids'][i]
            all_subject_ids = batch['all_subject_ids'][i]
            
            # Map question indices to subjects
            output_subjects = []
            for q_idx in output_q_indices[:10]:  # First 10
                # Find this q_idx in all_q_ids
                if q_idx < len(all_q_ids):
                    q_id = all_q_ids[q_idx]
                    # Find subject for this q_id
                    try:
                        pos = np.where(np.array(all_q_ids) == q_id)[0][0]
                        if pos < len(all_subject_ids):
                            output_subjects.append(all_subject_ids[pos])
                    except:
                        output_subjects.append("unknown")
            
            log(f"    Subjects in first 10 output questions: {output_subjects}")
            subject_155_in_output = sum(1 for s in output_subjects if isinstance(s, list) and 155 in s)
            log(f"    Questions with subject 155 in first 10 output: {subject_155_in_output}")
    
    log("\n" + "="*80)
    log(f"SUMMARY FOR {dataset_name}")
    log("="*80)
    log(f"\nTotal unique questions seen across {n_batches} batches:")
    log(f"  Input (trainable) questions: {len(all_input_questions)}")
    log(f"  Output (meta) questions: {len(all_output_questions)}")
    log(f"  All questions (union): {len(all_questions)}")
    log(f"  Out of 948 total questions: {100*len(all_questions)/948:.1f}%")
    
    log(f"\nAverage per student:")
    log(f"  Total questions: {np.mean(student_question_counts):.1f} ± {np.std(student_question_counts):.1f}")
    log(f"  Input questions: {np.mean(student_input_counts):.1f} ± {np.std(student_input_counts):.1f}")
    log(f"  Output questions: {np.mean(student_output_counts):.1f} ± {np.std(student_output_counts):.1f}")
    
    # Show most and least common questions
    sorted_questions = sorted(question_frequency.items(), key=lambda x: x[1], reverse=True)
    
    log(f"\nMost common questions (seen in most students):")
    for q_id, count in sorted_questions[:10]:
        log(f"  Q{q_id}: seen {count} times")
    
    log(f"\nQuestions never seen (in these {n_batches} batches):")
    unseen = set(range(948)) - all_questions
    log(f"  Count: {len(unseen)}")
    log(f"  IDs (first 20): {sorted(list(unseen))[:20]}")
    
    return all_questions, question_frequency


def simulate_active_sampling(dataset, dataset_name, n_students=10):
    # Simulate what happens during active sampling in training
    
    log("\n" + "="*80)
    log(f"SIMULATING ACTIVE SAMPLING: {dataset_name}")
    log("="*80)
    
    collate_fn_obj = collate_fn(n_question=948)
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        collate_fn=collate_fn_obj, 
        batch_size=32, 
        num_workers=0, 
        shuffle=False,
        drop_last=False
    )
    
    # Get first batch
    batch = next(iter(train_loader))
    
    log(f"\nAnalyzing first {min(n_students, len(batch['user_ids']))} students:")
    
    for i in range(min(n_students, len(batch['user_ids']))):
        log(f"\n--- Student {i} (user_id: {batch['user_ids'][i]}) ---")
        
        # What questions can be sampled from?
        available_mask = batch['input_mask'][i].clone()
        available_q_indices = torch.where(available_mask == 1)[0].cpu().numpy()
        
        log(f"Available questions for active sampling: {len(available_q_indices)}")
        log(f"  Question IDs (first 20): {available_q_indices[:20].tolist()}")
        log(f"  Question IDs (last 20): {available_q_indices[-20:].tolist()}")
        
        # Check distribution
        log(f"  Min question ID: {available_q_indices.min()}")
        log(f"  Max question ID: {available_q_indices.max()}")
        log(f"  Range coverage: Q{available_q_indices.min()} to Q{available_q_indices.max()}")
        
        # Check if questions are clustered or spread out
        gaps = np.diff(sorted(available_q_indices))
        log(f"  Average gap between questions: {gaps.mean():.1f}")
        log(f"  Max gap: {gaps.max()}")


def investigate_original_students(train_data, n_students=10):
    # Look at original student data before any dataset processing
    
    log("\n" + "="*80)
    log("INVESTIGATING ORIGINAL STUDENT DATA")
    log("="*80)
    
    # Filter students who have subject 155
    students_with_155 = []
    for student in train_data:
        count_155 = sum(1 for sids in student['subject_ids'] if 155 in sids)
        if count_155 >= 55:
            students_with_155.append(student)
    
    log(f"\nStudents with >=55 questions of subject 155: {len(students_with_155)}")
    
    # Analyze question distribution
    all_q_ids_in_filtered_students = set()
    q_id_frequency = Counter()
    
    for student in students_with_155[:n_students]:
        log(f"\nStudent {student['user_id']}:")
        log(f"  Total questions: {len(student['q_ids'])}")
        
        # Count by subject
        subject_counts = Counter()
        for sids in student['subject_ids']:
            for sid in sids:
                subject_counts[sid] += 1
        
        log(f"  Questions with subject 155: {subject_counts[155]}")
        log(f"  Total unique subjects: {len(subject_counts)}")
        log(f"  Top 5 subjects: {subject_counts.most_common(5)}")
        
        # Track q_ids
        for q_id in student['q_ids']:
            all_q_ids_in_filtered_students.add(q_id)
            q_id_frequency[q_id] += 1
        
        # Show range of q_ids
        q_ids_array = np.array(student['q_ids'])
        log(f"  Question ID range: {q_ids_array.min()} to {q_ids_array.max()}")
        log(f"  First 20 q_ids: {student['q_ids'][:20]}")
    
    log(f"\n" + "="*80)
    log(f"OVERALL STATISTICS (first {n_students} filtered students)")
    log("="*80)
    log(f"\nUnique question IDs seen: {len(all_q_ids_in_filtered_students)} out of 948")
    log(f"Coverage: {100*len(all_q_ids_in_filtered_students)/948:.1f}%")
    
    # Most common questions
    log(f"\nMost common questions:")
    for q_id, count in q_id_frequency.most_common(10):
        log(f"  Q{q_id}: appears in {count}/{n_students} students")
    
    # Questions never seen
    unseen = set(range(948)) - all_q_ids_in_filtered_students
    log(f"\nQuestions never seen (in these {n_students} students): {len(unseen)}")
    if len(unseen) <= 50:
        log(f"  IDs: {sorted(unseen)}")
    else:
        log(f"  IDs (first 20): {sorted(unseen)[:20]}")


if __name__ == "__main__":
    # Investigate original data
    log("\n" + "="*80)
    log("STEP 1: UNDERSTAND ORIGINAL DATA")
    log("="*80)
    
    train_dataset_filtered, train_dataset_unfiltered, train_data = investigate_dataset_filtering()
    
    # Look at original students
    investigate_original_students(train_data, n_students=20)
    
    # Analyze batches from filtered dataset
    log("\n\n")
    questions_filtered, freq_filtered = analyze_batches(
        train_dataset_filtered, 
        "FILTERED (meta_filtered=True)", 
        n_batches=10
    )
    
    # Analyze batches from unfiltered dataset
    log("\n\n")
    questions_unfiltered, freq_unfiltered = analyze_batches(
        train_dataset_unfiltered, 
        "UNFILTERED (meta_filtered=False)", 
        n_batches=10
    )
    
    # Compare coverage
    log("\n" + "="*80)
    log("COMPARING QUESTION COVERAGE")
    log("="*80)
    log(f"\nQuestions seen in filtered dataset: {len(questions_filtered)}")
    log(f"Questions seen in unfiltered dataset: {len(questions_unfiltered)}")
    log(f"Questions only in unfiltered: {len(questions_unfiltered - questions_filtered)}")
    log(f"Questions only in filtered: {len(questions_filtered - questions_unfiltered)}")
    
    # Simulate active sampling
    log("\n\n")
    simulate_active_sampling(train_dataset_filtered, "FILTERED", n_students=5)
    
    log("\n\n")
    simulate_active_sampling(train_dataset_unfiltered, "UNFILTERED", n_students=5)
    
    log("\n" + "="*80)
    log("INVESTIGATION COMPLETE")
    log("="*80)
    
    # Get filename before closing
    filename = output_file.name
    output_file.close()
    print(f"\n\nResults saved to {filename}")