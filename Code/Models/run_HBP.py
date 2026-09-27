from Models.HBP import ODL
from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
import time

def _prediction_without_graph(prediction):
    if hasattr(prediction, "detach"):
        return prediction.detach()
    return prediction

def _clear_model_step_cache(model):
    model.prediction.clear()
    if hasattr(model, "loss_array"):
        model.loss_array.clear()

def arch_change_create_param_list(model_params):
    params_list = []
    for max_num_hidden_layers in model_params['max_num_hidden_layers']:
        for qtd_neuron_per_hidden_layer in model_params['qtd_neuron_per_hidden_layer']:
            for b in model_params['b']:
                for s in model_params['s']:
                    for n in model_params['n']:
                        params_list.append({'max_num_hidden_layers': max_num_hidden_layers,
                            'qtd_neuron_per_hidden_layer': qtd_neuron_per_hidden_layer,
                            'b': b, 's': s, 'n': n, 'use_cuda': model_params['use_cuda'],
                            'batch_size': model_params['batch_size'],
                            'n_classes': model_params['n_classes'],
                        })
    return params_list

def run_HBP(Y, X_haphazard, mask, num_runs, model_params):
    result = {}
    params_list = arch_change_create_param_list(model_params)
    X_haphazard = np.nan_to_num(X_haphazard, copy=False)
    print("Number of experiments to run: ", len(params_list))
    for k in range(len(params_list)):
        params = params_list[k]
        print("Experiment number: ", k+1, "\nParams: \n", params)
        eval_list = []
        for j in range(num_runs):
            # Seeding for model
            seed_everything(j)
            Y_pred = np.empty(X_haphazard.shape[0], dtype=int)
            Y_logits = np.empty(X_haphazard.shape[0], dtype=float)
            start_time = time.time()
            model = ODL(X_haphazard.shape[1],params["max_num_hidden_layers"],params["qtd_neuron_per_hidden_layer"],
                        params["n_classes"],
                        params["batch_size"], params["b"],
                        params["n"], params["s"],
                        params["use_cuda"])
            for i in tqdm(range(0, X_haphazard.shape[0])):
                model.partial_fit(X_haphazard[i].reshape(1, X_haphazard.shape[1]),
                                Y[i].reshape(1))
                prediction = _prediction_without_graph(model.prediction[-1])
                Y_logits[i] = prediction[0, 1].item()
                Y_pred[i] = torch.argmax(prediction).item()
                _clear_model_step_cache(model)
            taken_time = time.time() - start_time
            del model
            eval_list.append(get_all_metrics(Y, Y_pred.reshape(-1, 1), Y_logits.reshape(-1, 1), taken_time))
            print("Run number: ", j+1, "\n Metrics: \n", eval_list[j])
        result[str(params)] = eval_list
    return result
