import numpy as np
import torch
from torch.utils import data
import torch
import random
from utils.utils import open_json, dump_json


class Dataset(data.Dataset):

	# to filter sid by STUDENT, enter a filter_id - adjust cutoff
		# to filter by META set, enter TRUE

	def __init__(self, data, filter_id=None, filter_by_meta_set=False, seed=None):
		self.data = data
		self.seed = seed

		if filter_id is not None:
			self.filter_id = filter_id if isinstance(filter_id, list) else [filter_id]
		else:
			self.filter_id = None

		self.filter_by_meta_set = filter_by_meta_set

		if self.filter_id:
			print("old data length:",len(self.data))

			self.data = self._filter_data(self.data, self.filter_id, 55)

			print(f"data with more than 55 qs of subject id = {self.filter_id}:",len(self.data))

		# we have filtered by students with more than [threshold] qs of a subject id

		# print(f"data: {self.data[0]}")

		# qs_by_sid = self._get_answered_qs_by_sid(subject_dict, subj_id)
		# self.data = self._filter_data(self.data, qs_by_sid, count)


	def _filter_data(self, data, s_ids, cutoff = 55):
		print("filtering data by", s_ids)

		filtered_students = []
		
		for student in data:
			check = self._check_sid(student, s_ids, cutoff)
			if check: 
				filtered_students.append(student)
		# go through each student in data
			# _check_sid(student)
			# if true, add to output

		return filtered_students

	def __len__(self):
		return len(self.data)

	def _check_sid(self, student_data, sids, cutoff = 55):
		# checks how many times subject id appears in this student
		count = 0
		q_ids = []
		
		for i, ids in enumerate(student_data['subject_ids']):
			if any(sid in ids for sid in sids):
				q_ids.append(student_data["q_ids"][i])
				count += 1 
		
		return count >= cutoff

	# Normal
	def __getitem__(self, index):
		# return self.data[index]
		# filter -> make sure '5' of subj id is in the meta set, 
		'Generates one sample of data'

		data = self.data[index]
		if self.filter_by_meta_set and self.filter_id:
			return self._getitem_filter(data, index)

		observed_index = np.array([idx for idx in range(len(data['q_ids']))])
		if not self.seed:
			np.random.shuffle(observed_index)
		else:
			random.Random(index+self.seed).shuffle(observed_index)
		
		N = len(observed_index)
		# Ensure exactly 40 in meta set, rest in trainable
		target_size = 40
		if N < target_size:
			# If not enough samples, use all available for target
			target_size = N
		
		target_index = observed_index[-target_size:]
		trainable_index = observed_index[:-target_size]

		# input_ans = data['ans'][trainable_index]
		input_label = data['labels'][trainable_index]
		input_question = data['q_ids'][trainable_index]
		output_label = data['labels'][target_index]
		output_question = data['q_ids'][target_index]

		# print("Not filtered stats: ")
		# print(len(input_label))
		# print(len(input_question))
		# print(len(output_label))
		# print(len(output_question))

		# print(f"DEBUG __getitem__:")
		# print(f"  input_question (first 20): {input_question[:20]}")
		# print(f"  all_q_ids (first 20): {data['q_ids'][:20]}")

		# added q ids, subject ids, for this user
		output = {'input_label': torch.FloatTensor(input_label), 'input_question': torch.FloatTensor(input_question),
					'output_question': torch.FloatTensor(output_question), 'output_label': torch.FloatTensor(output_label),
					'user_id': data['user_id'],  'all_q_ids': data['q_ids'],  'all_subject_ids': data['subject_ids']}
		# 'input_ans': torch.FloatTensor(input_ans)
		return output
		

	# meta set filter
	def _getitem_filter(self, data, index, meta_set_length = 40):
		# return self.data[index]
		# filter -> make sure '5' of subj id is in the meta set, 
		'Generates one sample of data'

		observed_index = np.array([idx for idx in range(len(data['q_ids']))])
		if not self.seed:
			np.random.shuffle(observed_index)
		else:
			random.Random(index+self.seed).shuffle(observed_index)
		
		input_label = []
		input_question = []
		output_label = []
		output_question = []
		
		reached_threshold = False

		for i in observed_index:
			question, subject_ids = data["q_ids"][i], data["subject_ids"][i]
			if any(fid in subject_ids for fid in self.filter_id) and not reached_threshold:
				output_label.append(data["labels"][i])
				output_question.append(data["q_ids"][i])
				if len(output_label) == meta_set_length:
					reached_threshold = True
			else:
				input_label.append(data["labels"][i])
				input_question.append(data["q_ids"][i])

		# print(f"DEBUG _getitem_filter:")
		# print(f"  input_question (first 20): {input_question[:20]}")
		# print(f"  all_q_ids (first 20): {data['q_ids'][:20]}")

		# print("Filtered stats: ")
		# print(len(input_label))
		# print(len(input_question))
		# print(len(output_label))
		# print(len(output_question))

		# input_ans = data['ans'][trainable_index]
		# input_label = data['labels'][trainable_index]
		# input_question = data['q_ids'][trainable_index]
		# output_label = data['labels'][target_index]
		# output_question = data['q_ids'][target_index]

		input_label = np.array(input_label)
		input_question = np.array(input_question)
		output_label = np.array(output_label)
		output_question = np.array(output_question)

		# added q ids, subject ids, for this user
		output = {'input_label': torch.FloatTensor(input_label), 'input_question': torch.FloatTensor(input_question),
					'output_question': torch.FloatTensor(output_question), 'output_label': torch.FloatTensor(output_label),
					'user_id': data['user_id'],  'all_q_ids': data['q_ids'],  'all_subject_ids': data['subject_ids'],
					'enough_sids': self._check_sid(data, sids=self.filter_id) }
		# 'input_ans': torch.FloatTensor(input_ans)
		return output

	# def __getitem__(self, index):
	# 	# return self.data[index]
	# 	# filter -> make sure '5' of subj id is in the meta set, 
	# 	'Generates one sample of data'
	# 	data = self.data[index]
	# 	print("generating sample from", data)
	# 	observed_index = np.array([idx for idx in range(len(data['q_ids']))])
	# 	if not self.seed:
	# 		np.random.shuffle(observed_index)
	# 	else:
	# 		random.Random(index+self.seed).shuffle(observed_index)
	# 	N = len(observed_index)
	# 	target_index = observed_index[-N//5:]
	# 	trainable_index = observed_index[:-N//5]

	# 	# input_ans = data['ans'][trainable_index]
	# 	input_label = data['labels'][trainable_index]
	# 	input_question = data['q_ids'][trainable_index]
	# 	output_label = data['labels'][target_index]
	# 	output_question = data['q_ids'][target_index]

	# 	# added q ids, subject ids, for this user
	# 	output = {'input_label': torch.FloatTensor(input_label), 'input_question': torch.FloatTensor(input_question),
	# 				'output_question': torch.FloatTensor(output_question), 'output_label': torch.FloatTensor(output_label),
	# 				'user_id': data['user_id'],  'all_q_ids': data['q_ids'],  'all_subject_ids': data['subject_ids'] }
	# 	# 'input_ans': torch.FloatTensor(input_ans)
	# 	return output


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
