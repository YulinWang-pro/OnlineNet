import random
import os
import numpy as np
import torch
import torch.nn.functional as F
import io
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

trans = transforms.ToTensor()


# --------------Seed--------------#
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况

        # 设置CuDNN确定性操作（可能会降低性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Prepare data for naive Bayes
def process_single_input(x, mask):
    return {f"{i}": val for i, val in enumerate(x) if mask[i]}


def prepare_data_naiveBayes(X, mask):
    return list(map(process_single_input, X, mask))


# Impute to create base features
def impute_data(data, mask, imputation_type):
    if imputation_type == 'forwardfill':
        arr = data.T
        mask = np.invert(mask.astype(bool))
        mask = mask.T
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = arr[np.arange(idx.shape[0])[:, None], idx]
        # If there is no information to impute (for the initial instances), impute by 0
        return out.T
    elif imputation_type == 'forwardmean':
        with np.errstate(divide='ignore', invalid='ignore'):
            forward_mean_arr = np.cumsum(data, axis=0) / np.cumsum(mask, axis=0)
        forward_mean_arr[np.isnan(forward_mean_arr)] = 0
        mask = np.invert(mask.astype(bool))
        data[mask] = forward_mean_arr[mask]
        return data
    elif imputation_type == 'zerofill':
        return data
    return data


# Prepare data for AuxNet (can also be used for AuxDrop)
def prepare_data_imputation(X_haphazard, mask, imputation_type, n_impute_feat):
    base_index_list = []
    i = 0
    while n_impute_feat > len(base_index_list):
        temp_index = np.where(mask[i] == 1)[0]
        i = i + 1
        base_index_list = list(set(base_index_list) | set(temp_index))

    base_index_list = base_index_list[:n_impute_feat]
    aux_index_list = [a for a in range(mask.shape[1]) if a not in base_index_list]
    X_base = impute_data(X_haphazard[:, base_index_list], mask[:, base_index_list], imputation_type)

    return X_base, X_haphazard[:, aux_index_list], mask[:, aux_index_list]


# Dummy feature creation for aux-drop and aux-net code
def dummy_feat(n_instances, n_feat, dummy_type="standardnormal"):
    if dummy_type == "standardnormal":
        return np.random.normal(0, 1, (n_instances, n_feat))


# ------------Min-Max Normalization----------------------#
# performs min max normalization on supplied data instance with dropped features
# requires running min and max for all features, returnss their updated values
# for the first entry in dataset min max will be intialized to an array consisting of nan values
def minmaxnorm(row, min_arr, max_arr, epsilon=1e-15):
    # Create boolean mask for non-nan values
    valid_mask = ~np.isnan(row)

    # Handle initial case for min_arr and max_arr
    min_mask = np.isnan(min_arr)
    max_mask = np.isnan(max_arr)
    min_arr = np.where(min_mask & valid_mask, row, min_arr)
    max_arr = np.where(max_mask & valid_mask, row, max_arr)

    # Update min and max arrays
    np.minimum(min_arr, row, out=min_arr, where=valid_mask)
    np.maximum(max_arr, row, out=max_arr, where=valid_mask)

    # Perform min-max normalization
    norm_row = np.full_like(row, np.nan)
    denominator = max_arr - min_arr + epsilon
    np.divide(row - min_arr, denominator, out=norm_row, where=valid_mask & (denominator != 0))

    return norm_row, min_arr, max_arr


def zscore(row, run_sum, sum_sq, count, epsilon=1e-30):
    mask = ~np.isnan(row)  # select visible, that is, non-nan values

    count += (
        mask)  # the list count maintains the count of instances encounterd for all features. Increment count for each visible feature

    run_sum = np.where(mask, run_sum + row, run_sum)  # update running sum for all non-nan features
    sum_sq = np.where(mask, sum_sq + row ** 2, sum_sq)  # update sum of squares for all non-nan features

    mean = np.where(mask, run_sum / (count + epsilon), np.nan)  # calculate mean for all non-nan features

    variance = sum_sq / (count + epsilon) - mean ** 2
    std = np.where(mask, np.sqrt(np.maximum(variance, 0)), np.nan) + epsilon
    #std = np.where(mask, np.sqrt(sum_sq / (count + epsilon) - mean ** 2),
                   #np.nan) + epsilon  # calculate standard deviation for all non-nan features. Add epsilon to avoid division by zero

    norm_row = np.where(mask, (row - mean) / (std + epsilon),
                        np.nan)  # calculate z-score normalized values for all non-nan features

    return norm_row, run_sum, sum_sq, count


# ------------------------- Nan Marking -----------------------#

# NAN VALUES REPRESENTED AS CROSSES
# function to produce image tensors corresponding to given normalized data instance. USES MATPLOTLIB AND HENCE IS VERY SLOW.
# takes normalized and dropped values, reverse mask, color scheme and dots per inch resolution
def bar_nan_mark_z_score_plot(values, rev_val, colors, feat, vert=True, dpi=56):
    s = 224 / dpi  # s is the size of image in inches. Models require height and width of image to be 224
    fig, _ = plt.subplots(
        figsize=(s, s))  # fix fig as s*s image. When saving with corresponding dpi we get 224*224 image
    fig = plt.bar(feat, values, color=colors)  # represnt the min-max normed features as bars
    fig = plt.scatter(feat, rev_val, s=180, color='red', marker='x')  # plot crosses at dropped instances
    fig = plt.ylim((-3, 3))  # set y axis limits so that all graphs are scaled uniformly
    fig = io.BytesIO()  # creates byte buffer for converting plt bar object to jpg

    figsav = plt.savefig(fig, dpi=dpi, format='jpg')  # saving to jpg

    image = trans(Image.open(fig))  # opening jpg using PIL and transforming to PyTorch tensor
    if not vert:
        image = transforms.functional.rotate(image, angle=270)
    plt.close()
    return image


# -----------------------------No-Nan Markings---------------------#
def bar_min_max_plot(values, colors, spacing=0.3):  # for min max normalized data
    # Convert inputs to PyTorch tensors if they're not already
    values = torch.tensor(values)
    colors = torch.tensor(colors)

    # Filter out NaN values and their corresponding colors
    valid_mask = ~torch.isnan(values)
    valid_values = values[valid_mask]
    valid_colors = colors[valid_mask]

    # Create a 3x224x224 RGB image with white background
    img = torch.ones((3, 224, 224), dtype=torch.float32)

    num_bars = len(valid_values)
    bar_width = int(224 / (num_bars + (num_bars + 1) * spacing))
    gap_width = int(bar_width * spacing)

    # Ensure we have at least 1 pixel for each bar and gap
    if bar_width == 0 or gap_width == 0:
        bar_width = max(1, int(224 / (2 * num_bars - 1)))
        gap_width = max(1, 224 - num_bars * bar_width) // (num_bars + 1)

    # Calculate positions for bars
    positions = torch.arange(num_bars) * (bar_width + gap_width) + gap_width

    # Calculate heights for bars
    heights = (valid_values * 224).int()

    # Create mask for all bars at once
    y_indices = torch.arange(224).unsqueeze(1).unsqueeze(2)
    x_indices = torch.arange(224).unsqueeze(0).unsqueeze(2)

    positions = positions.unsqueeze(0).unsqueeze(0)
    heights = heights.unsqueeze(0).unsqueeze(0)

    bar_masks = (x_indices >= positions) & (x_indices < (positions + bar_width)) & (y_indices >= (224 - heights))

    # Apply colors to bars
    for i, color in enumerate(valid_colors):
        img[:, bar_masks[:, :, i]] = color.float().unsqueeze(1)

    return img


def pie_min_max_plot(sizes, colors, max_sectors=1000):  # plots pie charts without using matplotlib.
    sizes = torch.Tensor(sizes)
    colors = torch.Tensor(colors)
    # Filter out NaN values and their corresponding colors
    valid_mask = ~torch.isnan(sizes)
    valid_sizes = sizes[valid_mask]
    valid_colors = colors[valid_mask].to(torch.float32)

    # Return blank image in the edge case that all features have 0 values
    if valid_sizes.sum() == 0:
        return torch.ones((3, 224, 224), dtype=torch.float32)

    num_sectors = len(valid_sizes)

    if num_sectors <= max_sectors:
        # Original method
        img = torch.ones((3, 224, 224), dtype=torch.float32)
        center = torch.tensor([112, 112])
        radius = 100

        total = valid_sizes.sum()
        angles = (valid_sizes / total) * (2 * torch.pi)
        cumulative_angles = torch.cumsum(angles, dim=0)

        y, x = torch.meshgrid(torch.arange(224), torch.arange(224), indexing='ij')
        x = x - center[0]
        y = y - center[1]

        r = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x)
        theta = torch.where(theta < 0, theta + 2 * torch.pi, theta)

        masks = [(r <= radius) & (theta >= prev_angle) & (theta < curr_angle)
                 for prev_angle, curr_angle in
                 zip(torch.cat([torch.tensor([0.]), cumulative_angles[:-1]]), cumulative_angles)]

        for mask, color in zip(masks, valid_colors):
            img[:, mask] = color.unsqueeze(1)

    else:
        # Resizing method for large number of sectors
        size = 1000
        img = torch.ones((3, size, size), dtype=torch.float32)
        center = torch.tensor([size / 2, size / 2])
        radius = size / 2 - 1

        # Sort and limit the number of sectors if it exceeds max_sectors
        sorted_indices = torch.argsort(valid_sizes, descending=True)
        valid_sizes = valid_sizes[sorted_indices[:max_sectors - 1]]
        valid_colors = valid_colors[sorted_indices[:max_sectors - 1]]
        others_size = sizes[sorted_indices[max_sectors - 1:]].sum()
        valid_sizes = torch.cat([valid_sizes, others_size.unsqueeze(0)])
        valid_colors = torch.cat([valid_colors, torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)])

        total = valid_sizes.sum()
        angles = (valid_sizes / total) * (2 * torch.pi)
        cumulative_angles = torch.cumsum(angles, dim=0)

        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        x = x - center[0]
        y = y - center[1]

        r = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x)
        theta = torch.where(theta < 0, theta + 2 * torch.pi, theta)

        for i, (start_angle, end_angle, color) in enumerate(
                zip(torch.cat([torch.tensor([0.]), cumulative_angles[:-1]]), cumulative_angles, valid_colors)):
            mask = (r <= radius) & (theta >= start_angle) & (theta < end_angle)
            img[:, mask] = color.unsqueeze(1)

        img = img.unsqueeze(0)
        img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        img = img.squeeze(0)

    return img


# plots values clipped in [-3,3] as bars. Positive values are plotted above x-axis and negative values below x-axis
def bar_z_score_plot(values, colors, spacing=0.3):
    # Convert inputs to PyTorch tensors if they're not already
    values = torch.tensor(values)
    colors = torch.tensor(colors, dtype=torch.float32)

    # Filter out NaN values and their corresponding colors
    valid_mask = ~torch.isnan(values)
    valid_values = values[valid_mask]
    valid_colors = colors[valid_mask]

    num_bars = len(valid_values)

    # Use original method if possible
    bar_width = int(224 / (num_bars + (num_bars + 1) * spacing))
    gap_width = int(bar_width * spacing)

    if bar_width >= 1 and gap_width >= 1:
        # Original method
        img = torch.ones((3, 224, 224), dtype=torch.float32)

        positions = torch.arange(num_bars) * (bar_width + gap_width) + gap_width

        # Calculate heights for positive and negative values separately
        heights = (torch.abs(valid_values) / 3 * 112).int().clamp(0, 112)

        y_indices = torch.arange(224).unsqueeze(1).unsqueeze(2)
        x_indices = torch.arange(224).unsqueeze(0).unsqueeze(2)

        positions = positions.unsqueeze(0).unsqueeze(0)
        heights = heights.unsqueeze(0).unsqueeze(0)

        # Create masks for positive and negative bars
        positive_mask = valid_values >= 0
        negative_mask = valid_values < 0

        positive_bar_masks = (x_indices >= positions) & (x_indices < (positions + bar_width)) & (
                y_indices >= (112 - heights)) & (y_indices < 112) & positive_mask
        negative_bar_masks = (x_indices >= positions) & (x_indices < (positions + bar_width)) & (y_indices >= 112) & (
                y_indices < (112 + heights)) & negative_mask

        for i, color in enumerate(valid_colors):
            img[:, positive_bar_masks[:, :, i]] = color.unsqueeze(1)
            img[:, negative_bar_masks[:, :, i]] = color.unsqueeze(1)

        # Add a horizontal line at y=0
        img[:, 111:113, :] = torch.tensor([0, 0, 0]).unsqueeze(1).unsqueeze(2).expand(3, 2, 224)  # Black line

    else:
        # Resizing method for large number of bars
        max_width = max(2000, num_bars * 2)  # Ensure at least 2 pixels per bar
        bar_width = max(1, int(max_width / (num_bars + (num_bars + 1) * spacing)))
        gap_width = max(1, int(bar_width * spacing))
        total_width = num_bars * bar_width + (num_bars + 1) * gap_width

        img = torch.ones((3, 224, total_width), dtype=torch.float32)
        positions = torch.arange(num_bars) * (bar_width + gap_width) + gap_width

        # Calculate heights for positive and negative values separately
        heights = (torch.abs(valid_values) / 3 * 112).int().clamp(0, 112)

        for i, (pos, height, color, value) in enumerate(zip(positions, heights, valid_colors, valid_values)):
            if value >= 0:
                img[:, 112 - height:112, pos:pos + bar_width] = color.unsqueeze(1).unsqueeze(2)
            else:
                img[:, 112:112 + height, pos:pos + bar_width] = color.unsqueeze(1).unsqueeze(2)

        # Add a horizontal line at y=0
        img[:, 111:113, :] = torch.tensor([0, 0, 0]).unsqueeze(1).unsqueeze(2).expand(3, 2, total_width)  # Black line

        img = img.unsqueeze(0)
        img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        img = img.squeeze(0)

    return img

