import torch
from torch.utils.data import DataLoader
import numpy as np
from utils.utils import compute_auc, compute_accuracy, data_split, batch_accuracy

def run_random_test():

	# run_random from train but keep track of the qs/subjects we are sampling

	# put it in some json: student_id: {q_ids: [], subject_ids: []}

	pass

# test model - while testing we want to keep track of questions / subjects sampled 
def test(model, dataset, config, params):
    model.eval()
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=params.test_batch_size,
        num_workers=params.num_workers,
        shuffle=False,
        drop_last=False
    )

    all_preds, all_targets = [], []
    n_batch = 0
    
	# for each batch, run_random_test (sample questions)
    for batch in loader:
        output = run_random_test(batch, config)
        
        target = batch['output_labels'].float().numpy()
        mask = batch['output_mask'].numpy() == 1

        all_preds.append(output[mask])
        all_targets.append(target[mask])
        n_batch += 1

    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)

    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return auc, accuracy

if __name__ == "__main__":

	# load model

	# load the test data that was resered from train.py - put in Dataset

	# batch

	# test_model()
	pass