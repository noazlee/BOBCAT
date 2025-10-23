import torch
from model import MAMLModel

def load_model():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoint = torch.load("BOBCAT/saved_models/bobcat_final_eedi-3_biirt-active_10.pt", map_location=device)

	model_params = checkpoint.get('params', {})

	model = MAMLModel(
        n_question=model_params.get('n_question'),        
        question_dim=model_params.get('question_dim', 1),
        dropout=model_params.get('dropout', 0.2),
        sampling=model_params.get('sampling', 'active'),
        n_query=model_params.get('n_query', 10),
    ).to(device)

	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()

	meta_params = checkpoint["meta_params"][0]
	meta_params = meta_params.to(device)

	return model, meta_params



def run_random_test():

	# run_random from train but keep track of the qs/subjects we are sampling

	# put it in some json: student_id: {q_ids: [], subject_ids: []}

	pass

# test model - while testing we want to keep track of questions / subjects sampled 
def test(model):

	# for each batch, run_random_test (sample questions)

	# metrics

	pass

# 

if __name__ == "__main__":

	# load model

	# load the test data that was resered from train.py - put in Dataset

	# batch

	# test_model()
