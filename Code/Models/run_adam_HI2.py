#from Models.HI2 import HI2
from Utils.utils import seed_everything
from Utils.metric_utils import get_all_metrics
from tqdm import tqdm
import numpy as np
import time
import torch
import torchvision
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import timm
from Utils import utils


def create_param_list(model_params):
    params_list = []
    for lr in model_params['lr']:
        for spacing in model_params['spacing']:
                params_list.append({'lr': lr,
                                    'spacing': spacing})
    return params_list


def gen_colors(n, seed=37):
    #np.random.seed(seed)

    def generate_colors(count):
        # Generate colors as integers between 0 and 255
        return np.random.randint(0, 256, size=(count, 3))

    # Generate initial set of colors
    colors_int = generate_colors(n)

    # Check for uniqueness and generate more if needed
    unique_colors_int = np.unique(colors_int, axis=0)
    while len(unique_colors_int) < n:
        additional_colors = generate_colors(n - len(unique_colors_int))
        colors_int = np.vstack((unique_colors_int, additional_colors))
        unique_colors_int = np.unique(colors_int, axis=0)

    # Convert to float and round to 3 decimal places
    unique_colors_float = np.round(unique_colors_int.astype(float) / 255, 3)

    return unique_colors_float[:n]



def run_adam_HI2(X, labels, drop_df, mask, num_runs, model_params):
    rev_mask = (1 - mask)
    mat_rev_mask = np.where(rev_mask, 0.5, np.nan)
    result = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    params_list = create_param_list(model_params)
    model_name="res34"#["res34", "vit_small"]
    plot_type="bar_z_score"#["pie_min_max", "bar_z_score", "bar_nan_mark_z_score", "bar_min_max"]
    vert=True#True or False,'whether plot orientation is vertical(True) or horizontal(False)'
    for k in range(len(params_list)):
        params = params_list[k]
        lr = params['lr']
        spacing = params['spacing']
        eval_list = []
        print("Experiment number: ", k + 1, "\nParams: \n", params)
        for run in range(num_runs):
            # Seeding for model
            seed_everything(run)
            colors=gen_colors(X.shape[1], run)
            num_inst = drop_df.shape[0]
            num_feats = drop_df.shape[1]
            # 3. 重新初始化模型，避免使用上一次实验训练好的权重
            if model_name == 'res34':
                model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
                model.fc = nn.Linear(model.fc.in_features, 1)
            elif model_name == 'vit_small':
                model = timm.create_model('vit_small_patch16_224', pretrained=True)
                model.head = nn.Linear(model.head.in_features, 1)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            model = model.to(device)

            # 4. 重新初始化所有的追踪列表和统计变量
            loss_history = []
            preds = []
            pred_logits = []
            true = []

            feat = np.arange(num_feats)
            min_arr = np.array([np.nan] * num_feats)
            max_arr = np.copy(min_arr)

            minmax_time = 0
            plot_time = 0
            model_time = 0
            predict_time = 0

            run_sum = np.zeros(num_feats)
            sum_sq = np.zeros(num_feats)
            count = np.zeros(num_feats)

            model.train()
            start_time = time.time()
            # 5. 内层训练/推理循环
            for k in tqdm(range(num_inst), desc=f"Run {run + 1}"):
                row = drop_df[k]  # 缺失的为nan
                rev = mat_rev_mask[k]  # 缺失的为0.5
                label = torch.tensor(labels[k])

                start1 = time.time()
                if plot_type in ['bar_z_score', 'bar_nan_mark_z_score']:
                    norm_row, run_sum, sum_sq, count = utils.zscore(row, run_sum, sum_sq, count)
                else:
                    norm_row, min_arr, max_arr = utils.minmaxnorm(row, min_arr, max_arr, epsilon=1e-15)
                minmax_time += (time.time() - start1)

                start2 = time.time()
                if plot_type == 'bar_nan_mark_z_score':
                    img = utils.bar_nan_mark_z_score_plot(norm_row, rev, colors, feat, vert, dpi=56)
                elif plot_type == 'bar_min_max':
                    img = utils.bar_min_max_plot(norm_row, colors, spacing)
                elif plot_type == 'pie_min_max':
                    img = utils.pie_min_max_plot(norm_row, colors)
                elif plot_type == 'bar_z_score':
                    img = utils.bar_z_score_plot(norm_row, colors)
                plot_time += (time.time() - start2)

                img, label = img.to(device), label.to(device)
                img = torch.reshape(img, (-1, 3, 224, 224))

                # 模型前向传播与反向传播
                with torch.no_grad():
                    start3 = time.time()
                optimizer.zero_grad()
                outputs = model(img)
                outputs = torch.reshape(outputs, (-1,))
                loss = criterion(outputs, label.float())
                loss.backward()
                loss_history.append(loss.item())
                optimizer.step()

                with torch.no_grad():
                    model_time += (time.time() - start3)
                    start4 = time.time()
                    predicted = torch.sigmoid(outputs)
                    predicted = torch.round(predicted)

                    pred_logits.append(outputs.to('cpu').item())
                    preds.append(predicted.to('cpu').item())
                    true.append(label.to('cpu').item())
                    predict_time += (time.time() - start4)
            taken_time = time.time() - start_time
            del model
            eval_list.append(get_all_metrics(labels, np.array(preds).reshape(-1, 1), np.array(pred_logits).reshape(-1, 1), taken_time))
            print("Run number: ", run + 1, "\n Metrics: \n", eval_list[run])
        result[str(params)] = eval_list
    return result