# --------------Libraries--------------#
import os
import argparse
import numpy as np
import sys
import pickle

sys.path.append('/code/DataCode/')
path_to_result = "./Results/"
# --------------Import Functions--------------#
from Utils import utils
from DataCode import data_load

from Models.run_olvf import run_olvf
from Models.run_orf3v import run_orf3v
from Models.run_olifl import run_olifl
from Models.run_auxdrop import run_auxdrop, run_auxdrop_arch_change
from Models.run_ovfm import run_ovfm
from Config import config_models
from Models.run_HBP import run_HBP
from Models.run_adam_HI2 import run_adam_HI2
from Models.run_MODL import run_MODL
from Models.run_OnlineNetTau025 import run_OnlineNetTau025

# --------------All Variables--------------#
if __name__ == '__main__':
    __file__ = os.path.abspath('')
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=2023, type=int, help='Seeding Number')
    parser.add_argument('--type', default="noassumption", type=str,
                        choices=["noassumption", "basefeatures", "bufferstorage","Transformer","BERT","MCRA"],
                        help='The type of the experiment.')

    # Data Variables
    parser.add_argument('--dataname', default="wpbc", type=str,
                        choices=["a_Synthetic_dataset","synthetic", "a8a", "magic04",
                                 "spambase", "krvskp", "svmguide3",  "german",
                                 "diabetes_f", "wbc", "australian", "wdbc",  "wpbc","elec","weather","spam","phishing","LUdata","a","Synthetic_dataset"],
                        help='The name of the data')
    parser.add_argument('--syndatatype', default="variable_p", type=str,
                        help="The type to create suitable synthetic dataset")
    parser.add_argument('--probavailable', default=0.5, type=float,
                        help="The probability of each feature being available to create synthetic data")
    parser.add_argument('--ifbasefeat', default=False, type=bool,
                        help="If base features are available")

    # Method Variables
    parser.add_argument('--methodname', default="MODL", type=str,
                        choices=["olvf", "olifl", "ovfm", "orf3v", "auxdrop", "HBP", "adam_HI2", "MODL", "OnlineNetTau025"],
                        help="The name of the method")
    parser.add_argument('--initialbuffer', default=0, type=int,
                        help="The storage size of initial buffer trainig")
    parser.add_argument('--ifimputation', default=False, type=bool,
                        help="If some features needs to be imputed")
    parser.add_argument('--imputationtype', default='forwardfill', type=str,
                        choices=['forwardfill', 'forwardmean', 'zerofill'],
                        help="The type of imputation technique to create base features")
    parser.add_argument('--nimputefeat', default=2, type=int,
                        help="The number of imputation features")
    parser.add_argument('--ifdummyfeat', default=False, type=bool,
                        help="If some dummy features needs to be created")
    parser.add_argument('--dummytype', default='standardnormal', type=str,
                        help="The type of technique to create dummy base features")
    parser.add_argument('--ndummyfeat', default=1, type=int,
                        help="The number of dummy features to create")
    parser.add_argument('--ifAuxDropNoAssumpArchChange', default=False, type=bool,
                        help="If the Aux-Drop architecture needs to be changed to handle no assumption")
    parser.add_argument('--nruns', default=5, type=int,
                        help="The number of times a method should runs. For navie Bayes, it would be 1 because it is a deterministic method.")
    args = parser.parse_args()

    seed = args.seed
    type = args.type
    data_name = args.dataname
    syn_data_type = args.syndatatype
    p_available = args.probavailable
    if_base_feat = args.ifbasefeat
    method_name = args.methodname
    initial_buffer = args.initialbuffer
    if_imputation = args.ifimputation
    imputation_type = args.imputationtype
    n_impute_feat = args.nimputefeat
    if_dummy_feat = args.ifdummyfeat
    dummy_type = args.dummytype
    n_dummy_feat = args.ndummyfeat
    if_auxdrop_no_assumption_arch_change = args.ifAuxDropNoAssumpArchChange
    n_runs = args.nruns

    data_name_list = []
    if data_name == "synthetic":
        data_name_list = ["wpbc", "ionosphere", "wdbc", "australian", "wbc", "diabetes_f", "german", "ipd",
                          "svmguide3", "krvskp", "spambase"]  # , "magic04", "a8a", "susy", "higgs"
    elif data_name == "a":
       data_name_list = [  "wdbc", "australian", "wbc", "diabetes_f", "german",
                         "svmguide3", "LUdata","krvskp", "spambase","magic04","a8a","elec","weather","spam","phishing"]
    elif data_name == "a_Synthetic_dataset":
       data_name_list = ["wdbc", "australian", "wbc", "diabetes_f", "german",
                         "svmguide3", "LUdata","krvskp", "spambase","magic04","a8a","elec","weather","spam","phishing","SEA_f1_2500_to_f2_2500_w1_total5000",
                          "SEA_f1_2500_to_f2_2500_w500_total5000",
                          "SEA_f1_15000_to_f2_25000_to_f3_35000_to_f4_w1_total50000",
                          "SEA_f1_15000_to_f2_25000_to_f3_35000_to_f4_w5000_total50000",
                          "STAGGER_f1_2500_to_f2_2500_w1_total5000",
                          "STAGGER_f1_2500_to_f2_2500_w500_total5000",
                          "STAGGER_f1_20000_to_f2_40000_to_f3_w1_total50000",
                          "STAGGER_f1_20000_to_f2_40000_to_f3_w5000_total50000",
                          "LED_f1_2500_to_f2_2500_w1_total5000",
                          "LED_f1_2500_to_f2_2500_w500_total5000",
                          "LED_f1_15000_to_f2_25000_to_f3_35000_to_f4_w1_total50000",
                          "LED_f1_15000_to_f2_25000_to_f3_35000_to_f4_w5000_total50000"
                          ]
    else:
        data_name_list = [data_name]

    for data_name in data_name_list:
        print("Data Name:", data_name)

        result_addr = path_to_result + type + "/" + method_name + "/" + data_name

        data_type = "Synthetic"

        param_dict = {"type": type, "data_name": data_name, "data_type": data_type,
                      "seed": seed, "method_name": method_name, "initial_buffer": initial_buffer,
                      "syn_data_type": syn_data_type,
                      "p_available": p_available, "if_base_feat": if_base_feat,
                      "if_imputation": if_imputation, "imputation_type": imputation_type,
                      "n_impute_feat": n_impute_feat, "if_dummy_feat": if_dummy_feat,
                      "dummy_type": dummy_type, "n_dummy_feat": n_dummy_feat, "n_runs": n_runs,
                      "if_auxdrop_no_assumption_arch_change": if_auxdrop_no_assumption_arch_change
                      }
        # --------------SeedEverything--------------#
        utils.seed_everything(seed)

        # --------------Load Data wrt Variables--------------#
        if data_type == "Synthetic":
            X, Y, X_haphazard, mask = data_load.data_load_synthetic(data_name, syn_data_type,
                                                                    p_available, if_base_feat)
            result_addr = result_addr + "_prob_" + str(int(p_available * 100)) + ".data"
        else:
            X, Y, X_haphazard, mask = data_load.data_load_real(data_name)
            result_addr = result_addr + ".data"

        # --------------Model Configs--------------#
        if method_name == "OnlineNetTau025":
            n_runs, param_dict["config"] = config_models.config_OnlineNetTau025(data_name)
            param_dict["n_runs"] = n_runs
        if method_name == "MODL":
            n_runs, param_dict["config"] = config_models.config_MODL(data_name)
            param_dict["n_runs"] = n_runs
        if method_name == "adam_HI2":
            n_runs, param_dict["config"] = config_models.config_HI2(data_name)
            param_dict["n_runs"] = n_runs
            X_haphazard = np.where(mask, X, np.nan)
        elif method_name == "olvf":
            n_runs, param_dict["config"] = config_models.config_olvf(data_name, X.shape[1])
            param_dict["n_runs"] = n_runs
        elif method_name == "olifl":
            param_dict["config"] = config_models.config_olifl(data_name)
        elif method_name == "ovfm":
            n_runs,param_dict["config"] = config_models.config_ovfm(data_name)
        elif method_name == "orf3v":
            n_runs,param_dict["config"] = config_models.config_orf3v(data_name)
        elif method_name == "auxdrop":
            if if_auxdrop_no_assumption_arch_change:
                n_runs,param_dict["config"] = config_models.config_auxdrop(if_auxdrop_no_assumption_arch_change,
                                                                    X, data_name, if_imputation, if_dummy_feat,
                                                                    n_dummy_feat, X_haphazard, mask, imputation_type,
                                                                    dummy_type)
            else:
                n_runs,param_dict["config"], X_base, X_aux_new, aux_mask = config_models.config_auxdrop(
                    if_auxdrop_no_assumption_arch_change,
                    X, data_name, if_imputation, if_dummy_feat,
                    n_dummy_feat, X_haphazard, mask, imputation_type,
                    dummy_type)
        elif method_name == "HBP":
                n_runs,param_dict["config"] = config_models.config_HBP(data_name)
                param_dict["n_runs"] = n_runs
        print(param_dict)
        # --------------Run Model--------------#
        result = []
        if method_name == "OnlineNetTau025":
            result = run_OnlineNetTau025(X_haphazard, mask, Y, n_runs, param_dict["config"])
        if method_name == "MODL":
            result = run_MODL(X_haphazard, mask, Y, n_runs, param_dict["config"])
        if method_name == "adam_HI2":
            result = run_adam_HI2(X, Y, X_haphazard, mask, n_runs, param_dict["config"])
        elif method_name == "olvf":
            result = run_olvf(X_haphazard, mask, Y, n_runs, param_dict["config"])
        elif method_name == "olifl":
            result = run_olifl(X, Y, X_haphazard, mask, n_runs, param_dict["config"])
        elif method_name == "ovfm":
            result = run_ovfm(X, Y, X_haphazard, mask, n_runs, param_dict["config"])
        elif method_name == "orf3v":
            result = run_orf3v(X, Y, X_haphazard, mask, n_runs, param_dict["config"], initial_buffer)
        elif method_name == "auxdrop":
            if if_auxdrop_no_assumption_arch_change:
                result = run_auxdrop_arch_change(Y, X_haphazard, mask, n_runs, param_dict["config"])
            else:
                result = run_auxdrop(X_base, X_aux_new, aux_mask, Y, n_runs, param_dict["config"])
        elif method_name == "HBP":
            result = run_HBP(Y, X_haphazard, mask, n_runs, param_dict["config"])
        print(result)
        result_dict = {"params": param_dict, "results": result}

        # --------------Store results and all variables--------------#
        directory, filename = os.path.split(result_addr)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(result_addr, 'wb') as file:
            pickle.dump(result_dict, file) 
