import optuna
import random 


def subseq_permutation_detection(_string):
    # max window size
    l = len(_string) // 2
    #for various window sizes
    for window_size in range(2, l+1):
        for i in range(len(_string) - window_size + 1):
            subseq = _string[i:i+window_size]
            for j in range(i+window_size, len(_string) - window_size + 1):
                if subseq == _string[j:j+window_size]:
                    return True
    return False


# TODO: use mergekit to get the layer count
model_to_layer_count = {
    "psmathur/orca_mini_v3_13b": 23,
    "garage-bAInd/Platypus2-13B": 24,
}


def generate_objective_function(model_to_layer_count, repetitions=2):
    _a =  model_to_layer_count.items()
    _a = [(_model, index_no) for _model, layer_num in _a for index_no in range(layer_num)]
    def objective(trial):
        _seq = []
        for i in range(len(_a)*repetitions):
            _seq.append(trial.suggest_categorical(f"layer_{i}", _a))
        
        if subseq_permutation_detection(_seq):
            raise optuna.TrialPruned()
        
        # throw to mergekit


        # eval on metric



    return objective






    return objective


r = 2 # number of allowable repetitions



def objective(trial):




# generate a 
