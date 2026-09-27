import argparse
import pickle
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', default="noassumption", type=str,
                        choices = ["noassumption", "basefeatures", "bufferstorage","Transformer","MCRA"],
                        help='The type of the experiment.')
    parser.add_argument('--dataname', default="wpbc", type=str,
                        choices=["a_Synthetic_dataset","synthetic", "a8a", "magic04",
                                 "spambase", "krvskp", "svmguide3",  "german",
                                 "diabetes_f", "wbc", "australian", "wdbc",  "wpbc","elec","weather","spam","phishing","LUdata","a","Synthetic_dataset"],
                        help='The name of the data')
    parser.add_argument('--probavailable', default = 0.5, type = float,
                        help = "The probability of each feature being available to create synthetic data")
    parser.add_argument('--methodname', default = "MODL", type = str,
                        choices = ["olvf", "olifl", "ovfm", "orf3v", "auxdrop", "HBP", "adam_HI2", "MODL", "OnlineNetTau025"],
                        help = "The name of the method")

    args = parser.parse_args()
    type = args.type
    data_name = args.dataname
    p_available = args.probavailable
    method_name = args.methodname

    data_name_list = []
    if data_name == "a":
       data_name_list = ["wdbc", "australian", "wbc", "diabetes_f", "german",
                         "svmguide3", "LUdata", "krvskp", "spambase", "magic04", "a8a", "elec", "weather", "spam",
                         "phishing"]
    elif data_name == "Synthetic_dataset":
       data_name_list = [ "SEA_f1_2500_to_f2_2500_w1_total5000",
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
                          "LED_f1_15000_to_f2_25000_to_f3_35000_to_f4_w5000_total50000"]
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

        path_to_result="./Results/"
        result_addr = path_to_result + type + "/" + method_name + "/" + data_name
        df_addr = result_addr

        data_type = "Synthetic"

        if data_type == "Synthetic":
            result_addr = result_addr + "_prob_" + str(int(p_available*100)) + ".data"
            df_addr = df_addr + "_prob_" + str(int(p_available*100))

        else:
            result_addr = result_addr + ".data"

        with open(result_addr, 'rb') as file:
            data = pickle.load(file)

        print("Parameters of this experiment:")
        print(data['params'], "\n")

        # print("All Results: ", pd.DataFrame(data['results']))

        # Calculate mean 
        index = []
        values_list = []
        for i in data['results'].keys():
            index.append(i)
            print(i, pd.DataFrame(data['results'][i]), pd.DataFrame(data['results'][i]).iloc[:, 5:6])
            val_list = pd.DataFrame(data['results'][i]).mean(axis = 0).values.tolist()
            values_list.append(['%.2f' % elem for elem in val_list])
        col_name = list(data['results'][i][0].keys())
        # print(index)
        
        mean_df = pd.DataFrame(values_list, index = index, columns=col_name)
        # print("Mean Values:")
        # print(mean_df)

        # Calculate std deviation 
        index = []
        values_list = []
        for i in data['results'].keys():
            index.append(i)
            val_list = pd.DataFrame(data['results'][i]).std(axis = 0).values.tolist()
            values_list.append(['%.2f' % elem for elem in val_list])
        col_name = list(data['results'][i][0].keys())
        # print(index)
        # print(values_list)

        std_df = pd.DataFrame(values_list, index = index, columns=col_name)
        # print("Std Values:")
        # print(std_df)

        if method_name == "olvf":
            final_df_file = df_addr + 'final.csv'
            print(method_name, ": ", mean_df)
            mean_df.to_csv(final_df_file)
        else:
            final_df = mean_df
            for i in col_name:
                final_df[i] = mean_df[i].astype(str) + '(' + std_df[i].astype(str) + ')'
            final_df_file = df_addr + 'final.csv'
            print(method_name, ": ", final_df)
            final_df.to_csv(final_df_file)
