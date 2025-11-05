import numpy as np
import torch
from torch.utils import data
import torch
import random
from utils.utils import open_json, dump_json


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data, seed=None):
        'Initialization'
        self.data = data
        self.seed = seed

        # qs_by_sid = self._get_answered_qs_by_sid(subject_dict, subj_id)
        # self.data = self._filter_data(self.data, qs_by_sid, count)


    def _get_answered_qs_by_sid(self, subject_dict, subj_id):
        # returns a list of all question ids / indices that have subject_dict answered in subj_id
        pass

    def _filter_data(qs_by_sid, count):
        # new list that only contains objects from the old list
        pass

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def _get_answered_qs_by_sid(self, sub_dict, sid):
        pass

    def __getitem__(self, index):
        # return self.data[index]
        # filter -> make sure '5' of subj id is in the meta set, 
        'Generates one sample of data'
        data = self.data[index]
        observed_index = np.array([idx for idx in range(len(data['q_ids']))])
        if not self.seed:
            np.random.shuffle(observed_index)
        else:
            random.Random(index+self.seed).shuffle(observed_index)
        N = len(observed_index)
        target_index = observed_index[-N//5:]
        trainable_index = observed_index[:-N//5]

        # input_ans = data['ans'][trainable_index]
        input_label = data['labels'][trainable_index]
        input_question = data['q_ids'][trainable_index]
        output_label = data['labels'][target_index]
        output_question = data['q_ids'][target_index]

        # added q ids, subject ids, for this user
        output = {'input_label': torch.FloatTensor(input_label), 'input_question': torch.FloatTensor(input_question),
                  'output_question': torch.FloatTensor(output_question), 'output_label': torch.FloatTensor(output_label),
                  'user_id': data['user_id'],  'all_q_ids': data['q_ids'],  'all_subject_ids': data['subject_ids'] }
        # 'input_ans': torch.FloatTensor(input_ans)
        return output


class collate_fn(object):
    def __init__(self, n_question):
        self.n_question = n_question

    def __call__(self, batch):
        B = len(batch)
        input_labels = torch.zeros(B, self.n_question).long()
        output_labels = torch.zeros(B, self.n_question).long()
        #input_ans = torch.ones(B, self.n_question).long()
        input_mask = torch.zeros(B, self.n_question).long()
        output_mask = torch.zeros(B, self.n_question).long()

        user_ids = []
        all_q_ids = []
        all_subject_ids = []

        for b_idx in range(B):
            input_labels[b_idx, batch[b_idx]['input_question'].long(
            )] = batch[b_idx]['input_label'].long()
            #input_ans[b_idx, batch[b_idx]['input_question'].long()] = batch[b_idx]['input_ans'].long()
            input_mask[b_idx, batch[b_idx]['input_question'].long()] = 1
            output_labels[b_idx, batch[b_idx]['output_question'].long(
            )] = batch[b_idx]['output_label'].long()
            output_mask[b_idx, batch[b_idx]['output_question'].long()] = 1

            # collecting metadata
            user_ids.append(batch[b_idx]['user_id'])
            all_q_ids.append(batch[b_idx]['all_q_ids'])
            all_subject_ids.append(batch[b_idx]['all_subject_ids'])


        output = {'input_labels': input_labels,  'input_mask': input_mask,
                  'output_labels': output_labels, 'output_mask': output_mask,
                  'user_ids': user_ids, 'all_q_ids': all_q_ids, 'all_subject_ids': all_subject_ids}
        # 'input_ans':input_ans,
        return output
