import numpy as np
import torch
import os
from dataset2 import Dataset, collate_fn
from utils.utils import compute_auc, compute_accuracy, data_split, batch_accuracy
from model import MAMLModel
from policy import PPO, Memory, StraightThrough
from copy import deepcopy
from utils.configuration import create_parser, initialize_seeds
import time
import os
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

# keeping track of questions sampled
import csv
from pathlib import Path

DEBUG = False if torch.cuda.is_available() else True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_val_score, best_test_score = 0, 0
best_val_auc, best_test_auc = 0, 0
best_epoch = -1


def clone_meta_params(batch):
    return [meta_params[0].expand(len(batch['input_labels']),  -1).clone(
    )]


def inner_algo(batch, config, new_params, create_graph=False):
    for _ in range(params.inner_loop):
        config['meta_param'] = new_params[0]
        res = model(batch, config)
        loss = res['train_loss']    
        grads = torch.autograd.grad(
            loss, new_params, create_graph=create_graph)
        new_params = [(new_params[i] - params.inner_lr*grads[i])
                      for i in range(len(new_params))]
        del grads
    config['meta_param'] = new_params[0]
    return


def get_rl_baseline(batch, config):
    model.pick_sample('random', config)
    new_params = clone_meta_params(batch)
    inner_algo(batch, config, new_params)
    with torch.no_grad():
        output = model(batch, config)['output']
    random_baseline = batch_accuracy(output, batch)
    return random_baseline


def pick_rl_samples(batch, config):
    env_states = model.reset(batch)
    action_mask, train_mask = env_states['action_mask'], env_states['train_mask']
    for _ in range(params.n_query):
        with torch.no_grad():
            state = model.step(env_states)
        if config['mode'] == 'train':
            actions = ppo_policy.policy_old.act(state, memory, action_mask)
        else:
            with torch.no_grad():
                actions = ppo_policy.policy_old.act(state, memory, action_mask)
        action_mask[range(len(action_mask)), actions], train_mask[range(
            len(train_mask)), actions] = 0, 1
        env_states['train_mask'], env_states['action_mask'] = train_mask, action_mask
    # train_mask
    config['train_mask'] = env_states['train_mask']
    return


def run_unbiased(batch, config):
    new_params = clone_meta_params(batch)
    config['available_mask'] = batch['input_mask'].to(device).clone()
    if config['mode'] == 'train':
        random_baseline = get_rl_baseline(batch, config)
    pick_rl_samples(batch, config)
    optimizer.zero_grad()
    meta_params_optimizer.zero_grad()
    inner_algo(batch, config, new_params)
    if config['mode'] == 'train':
        res = model(batch, config)
        loss = res['loss'] # ONLY META LOSS
        loss.backward()
        optimizer.step()
        meta_params_optimizer.step()
        ####
        final_accuracy = batch_accuracy(res['output'], batch)
        reward = final_accuracy - random_baseline
        memory.rewards.append(reward.to(device))
        ppo_policy.update(memory)
        #
    else:
        with torch.no_grad():
            res = model(batch, config)
    memory.clear_memory()
    return res['output']

# LOOK INTO THIS FOR  LOGGING BIASED
def pick_biased_samples(batch, config):
    new_params = clone_meta_params(batch)
    env_states = model.reset(batch)
    action_mask, train_mask = env_states['action_mask'], env_states['train_mask']
    for _ in range(params.n_query):
        with torch.no_grad():
            state = model.step(env_states)
            train_mask = env_states['train_mask']
        if config['mode'] == 'train':
            train_mask_sample, actions = st_policy.policy(state, action_mask)
        else:
            with torch.no_grad():
                train_mask_sample, actions = st_policy.policy(
                    state, action_mask)
        action_mask[range(len(action_mask)), actions] = 0
        # env state train mask should be detached
        env_states['train_mask'], env_states['action_mask'] = train_mask + \
            train_mask_sample.data, action_mask
        # train_mask_sample.data = question we are asking?
        if config['mode'] == 'train':
            # loss computation train mask should flow gradient
            config['train_mask'] = train_mask_sample+train_mask
            inner_algo(batch, config, new_params, create_graph=True)
            res = model(batch, config)
            loss = res['loss']
            st_policy.update(loss)
        
        # if test:
            # config['user_to_questions'] = []
            # .append(train_mask_sample.data)
            # also get the subject data from the q - make a map of q: set(subject_ids)
        
    config['train_mask'] = env_states['train_mask']
    
    return


def run_biased(batch, config):
    new_params = clone_meta_params(batch)
    if config['mode'] == 'train':
        model.eval()
    pick_biased_samples(batch, config)
    optimizer.zero_grad()
    meta_params_optimizer.zero_grad()
    inner_algo(batch, config, new_params)
    if config['mode'] == 'train':
        model.train()
        optimizer.zero_grad()
        res = model(batch, config)
        loss = res['loss']
        # loss = 0.9 * res['loss'] + 0.1 * res['train_loss']
        loss.backward()
        optimizer.step()
        meta_params_optimizer.step()
        ####
    else:
        with torch.no_grad():
            res = model(batch, config)
    return res['output']


def run_random(batch, config, batch_idx=None):

    new_params = clone_meta_params(batch)
    meta_params_optimizer.zero_grad()
    if config['mode'] == 'train':
        optimizer.zero_grad()
    ###
    config['available_mask'] = batch['input_mask'].to(device).clone()
    config['train_mask'] = torch.zeros(
        len(batch['input_mask']), params.n_question).long().to(device)

    # Random pick once
    config['meta_param'] = new_params[0]
    if sampling == 'random':
        model.pick_sample('random', config)
        inner_algo(batch, config, new_params)
    if sampling == 'active':
        for _ in range(params.n_query):
            model.pick_sample('active', config)
            inner_algo(batch, config, new_params)


    if config['mode'] == 'train':
        res = model(batch, config)
        # trainloss = res['train_loss']
        # trainloss.backward()
        loss = res['loss'] # ONLY META LOSS - WE DONT LEARN ANYTHING OTHER THAN SUBJECT 155 QUESTIONS
        # Weight the losses appropriately: loss = res['loss'] + 0.5 * res['train_loss']  
        loss.backward()
        optimizer.step()
        meta_params_optimizer.step()
        return
    else:
        with torch.no_grad():
            res = model(batch, config)
        output = res['output']
        return output


def train_model():
    global best_val_auc, best_test_auc, best_val_score, best_test_score, best_epoch
    config['mode'] = 'train'
    config['epoch'] = epoch
    # print("CONFIG:", config.keys()) # mode, epoch, available_mask, train_mask, meta_param
    # The model uses config to know:
    # - Whether to compute training loss or just predictions
    # - Which questions to use (via train_mask)
    # - Current meta-parameters
    model.train()
    N = [idx for idx in range(100, 100+params.repeat)]
    # print("N", N)
    for batch_id, batch in enumerate(train_loader): 
        # batch = { input_labels: , input_mask: , output_labels: , output_mask: , output_labels: , user_ids, all_q_ids, all_subject_ids }
        #            [512 x 948]    [512, 948]      [512, 948]      [512, 948]                       [512]   [512x[_]]     [512x[ [_] ]]

        # ------- DEBUGGING START HERE --------------------------

        # print("=" * 60)
        # print(f"Batch {batch_id}")

        # # Basic batch info
        # print(f"Batch keys: {batch.keys()}")
        # print(f"Batch size (num students): {len(batch['user_ids'])}")
        # print(f"Number of questions: {batch['input_labels'].shape[1]}")
        
        # # Shape information
        # print(f"\nTensor shapes:")
        # print(f"  input_labels shape: {batch['input_labels'].shape}")
        # print(f"  input_mask shape: {batch['input_mask'].shape}")
        # print(f"  output_labels shape: {batch['output_labels'].shape}")
        # print(f"  output_mask shape: {batch['output_mask'].shape}")
        
        # # User IDs
        # print(f"\nUser IDs in batch: {batch['user_ids'][:5]}...")  # First 5
        
        # # Look at first student's data
        # print(f"\n--- First Student (index 0) ---")
        # student_idx = 0
        
        # # Count questions for first student
        # input_questions_count = batch['input_mask'][student_idx].sum().item()
        # output_questions_count = batch['output_mask'][student_idx].sum().item()
        # print(f"Training questions: {input_questions_count}")
        # print(f"Test questions: {output_questions_count}")
        # print(f"Total questions answered: {input_questions_count + output_questions_count}")
        
        # # Which questions did they answer?
        # input_question_indices = torch.where(batch['input_mask'][student_idx] == 1)[0]
        # output_question_indices = torch.where(batch['output_mask'][student_idx] == 1)[0]
        # print(f"\nTraining question indices: {input_question_indices[:10].tolist()}...")  # First 10 - this matches!
        # print(f"Test question indices: {output_question_indices[:10].tolist()}...")
        
        # # What were their answers?
        # student_input_answers = batch['input_labels'][student_idx][input_question_indices[:10]]
        # print(f"\nFirst 10 training answers (0=wrong, 1=correct): {student_input_answers.tolist()}")
        
        # # Original question IDs for this student
        # print(f"\nOriginal question IDs for student {student_idx}:")
        # print(f"  Total questions in all_q_ids: {len(batch['all_q_ids'][student_idx])}")
        # print(f"  First 10 q_ids: {batch['all_q_ids'][student_idx][:10]}")
        
        # # Subject IDs
        # print(f"\nSubject IDs structure for student {student_idx}:")
        # print(f"  Type: {type(batch['all_subject_ids'][student_idx])}")
        # if len(batch['all_subject_ids'][student_idx]) > 0:
        #     print(f"  First few subjects: {batch['all_subject_ids'][student_idx][:5]}")


        # ------------- DEBUGGING END HERE --------------------------

        # Select RL Actions, save in config
        if sampling == 'unbiased':
            run_unbiased(batch, config)
        elif sampling == 'biased':
            run_biased(batch, config)
        else:
            run_random(batch, config)

    # Validation
    val_scores, val_aucs = [], []
    test_scores, test_aucs = [], []
    for idx in N:
        _, auc, acc = test_model(id_=idx, split='val')
        val_scores.append(acc)
        val_aucs.append(auc)
    val_score = sum(val_scores)/(len(N)+1e-20)
    val_auc = sum(val_aucs)/(len(N)+1e-20)

    if best_val_score < val_score:
        best_epoch = epoch
        best_val_score = val_score
        best_val_auc = val_auc
        # Run on test set
        for idx in N:
            _, auc, acc = test_model(id_=idx, split='test') # add function here - cli param: load existing model - test or test.py that loads existing model - save/load pickle
            test_scores.append(acc)
            test_aucs.append(auc)
        best_test_score = sum(test_scores)/(len(N)+1e-20)
        best_test_auc = sum(test_aucs)/(len(N)+1e-20)
    #
    print('Test_Epoch: {}; val_scores: {}; val_aucs: {}; test_scores: {}; test_aucs: {}'.format(
        epoch, val_scores, val_aucs, test_scores, test_aucs))
    if params.neptune:
        run["metrics/valid_accuracy"].append(val_score)
        run["metrics/best_test_accuracy"].append(best_test_score)
        run["metrics/best_test_auc"].append(best_test_auc)
        run["metrics/best_valid_accuracy"].append(best_val_score)
        run["metrics/best_valid_auc"].append(best_val_auc)
        run["metrics/best_epoch"].append(best_epoch)
        run["metrics/epoch"].append(epoch)


def test_model(id_, split='val'):
    model.eval()
    config['mode'] = 'test'
    if split == 'val':
        valid_dataset.seed = id_
    elif split == 'test':
        test_dataset.seed = id_
    loader = torch.utils.data.DataLoader(
        valid_dataset if split == 'val' else test_dataset, collate_fn=collate_fn, batch_size=params.test_batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    total_loss, all_preds, all_targets = 0., [], []
    n_batch = 0
    for batch in loader:
        if sampling == 'unbiased':
            output = run_unbiased(batch, config)
        elif sampling == 'biased':
            output = run_biased(batch, config)
        else:
            output = run_random(batch, config)
        target = batch['output_labels'].float().numpy()
        mask = batch['output_mask'].numpy() == 1
        all_preds.append(output[mask])
        all_targets.append(target[mask])
        n_batch += 1

    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    return total_loss/n_batch, auc, accuracy


if __name__ == "__main__":
    params = create_parser()
    print(params)
    print(params.sid_filtered)
    if params.use_cuda:
        assert device.type == 'cuda', 'no gpu found!'

    meta_filtered = True if params.meta_filtered else False

    if params.neptune:
        import neptune
        project = "noazlee-workspace/BOBCAT"
        run = neptune.init_run(  
            project=project,
            api_token=os.environ["NEPTUNE_API_TOKEN"],
            name=f"{params.sid_filtered_str}, {meta_filtered}, {params.model},{params.n_query},{params.dataset}",
        )
        run["parameters"] = vars(params)

    config = {}
    initialize_seeds(params.seed)

    #
    base, sampling = params.model.split('-')[0], params.model.split('-')[-1]
    if base == 'biirt':
        model = MAMLModel(sampling=sampling, n_query=params.n_query,
                          n_question=params.n_question, question_dim=1).to(device)
        meta_params = [torch.Tensor(
            1, 1).normal_(-1., 1.).to(device).requires_grad_()]
    if base == 'binn': # try binn instead
        model = MAMLModel(sampling=sampling, n_query=params.n_query,
                          n_question=params.n_question, question_dim=params.question_dim).to(device)
        meta_params = [torch.Tensor(
            1, params.question_dim).normal_(-1., 1.).to(device).requires_grad_()]

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=1e-8)
    meta_params_optimizer = torch.optim.SGD(
        meta_params, lr=params.meta_lr, weight_decay=2e-6, momentum=0.9)
    if params.neptune:
        run["model/summary"] = repr(model)
    print(model)

    #
    if sampling == 'unbiased':
        betas = (0.9, 0.999)
        K_epochs = 4                # update policy for K epochs
        eps_clip = 0.2              # clip parameter for PPO
        memory = Memory()
        ppo_policy = PPO(params.n_question, params.n_question,
                         params.policy_lr, betas, K_epochs, eps_clip)
        if params.neptune:
            run["model/ppo_summary"] = repr(ppo_policy.policy)
    if sampling == 'biased':
        betas = (0.9, 0.999)
        st_policy = StraightThrough(params.n_question, params.n_question,
                                    params.policy_lr, betas)
        if params.neptune:
            run["model/biased_summary"] = repr(st_policy.policy)

    #
    data_path = os.path.normpath('data/train_task_'+params.dataset+'.json')
    train_data, valid_data, test_data = data_split(
        data_path, params.fold,  params.seed)
    print("train data:",train_data[0], len(train_data)) # 2952 - user_id, subject_ids: [[], []], q_ids: nparray, labels: nparray
    print("valid data:",valid_data[0], len(valid_data)) # 983
    print("test data:",test_data[0], len(test_data))    # 983 - Save students here!
    if params.sid_filtered:
        # Training: NO filtering, model sees all questions 
        # 100 random - filter on meta set, teaching it what kind of thing it wants to predict
        train_dataset = Dataset(train_data, filter_id=params.sid_filtered, filter_by_meta_set=params.meta_filtered)
        # Validation/Test: WITH filtering, ensures subject 155 in meta set
        valid_dataset = Dataset(valid_data, params.sid_filtered, filter_by_meta_set=True) # Filter meta set for all - make 40 met aset for all
        test_dataset = Dataset(test_data, params.sid_filtered, filter_by_meta_set=True)   # fix control comparison, rerun with true/false
    else:
        train_dataset = Dataset(train_data)
        valid_dataset = Dataset(valid_data)
        test_dataset = Dataset(test_data)

    # FILTERING BY PEOPLE WITH SUBJECT ID = 155 - 665 train, 214 valid, 215 test
    
    #
    num_workers = 3
    collate_fn = collate_fn(params.n_question)
    train_loader = torch.utils.data.DataLoader( # batching users - efficiency, gradient stability, regulatization effect
        train_dataset, collate_fn=collate_fn, batch_size=params.train_batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    start_time = time.time()

    for epoch in range(params.n_epoch): #CHANGE LATER
        train_model()
        if epoch >= (best_epoch+params.wait):
            break

    
    # Save final model
    if not os.path.exists('saved_models_exp1'): 
        os.makedirs('saved_models_exp1') 

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if params.sid_filtered:
        model_save_path = f"saved_models_exp1/bobcat_final_{params.sid_filtered_str}_{params.meta_filtered}_{params.dataset}_{params.model}_{params.n_query}_{timestamp}.pt" 
    else:
        model_save_path = f"saved_models_exp1/bobcat_final_{params.dataset}_{params.model}_{params.n_query}_{timestamp}.pt" 

    torch.save({ 'epoch': epoch, 
        'model_state_dict': model.state_dict(), 
        'meta_params': meta_params, 
        'optimizer_state_dict': optimizer.state_dict(), 
        'meta_optimizer_state_dict': meta_params_optimizer.state_dict(), 
        'val_score': best_val_score, 
        'val_auc': best_val_auc, 
        'test_score': best_test_score, 
        'test_auc': best_test_auc, 
        'params': vars(params) 
        # 'test_qids: [] list of students we will test on
    }, model_save_path) 
    
    print(f"Final model saved to {model_save_path}")