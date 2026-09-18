import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from google.colab import drive
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

# Mount Drive early for checkpointing
drive.mount('/content/drive')
save_path = '/content/drive/My Drive/PandasCSVsaves'
os.makedirs(save_path, exist_ok=True)

# Set current_run_identifier to a string (e.g., 'experiment_A') to save to a new CSV,
# or leave None to use the default checkpoint file.
current_run_identifier = "RandseedV1_Optimized"

base_filename = 'detailed_classification_results'
if current_run_identifier:
    checkpoint_filename = f'{base_filename}_{current_run_identifier}.csv'
else:
    checkpoint_filename = f'{base_filename}.csv'

checkpoint_file = os.path.join(save_path, checkpoint_filename)

# Helper functions
def slice_trial(trial_data, window_size, step_size):
    chunks = []
    start_index = 0
    end_index = start_index + window_size
    total_length = trial_data.shape[1]
    while end_index <= total_length:
        chunk = trial_data[:, start_index:end_index]
        chunks.append(chunk)
        start_index += step_size
        end_index = start_index + window_size
    return np.array(chunks)

def majority_vote(predictions):
    values, counts = np.unique(predictions, return_counts=True)
    return values[np.argmax(counts)]

def slice_trials(X_train, y_train, window_size, step_size):
    new_X = []
    new_y = []
    for trial, label in zip(X_train, y_train):
        trial_chunks = slice_trial(trial, window_size, step_size)
        new_X.append(trial_chunks)
        new_y.extend([label] * trial_chunks.shape[0])
    if not new_X:
        return np.array([]), np.array([])
    X_train_sliced = np.vstack(new_X)
    y_train_sliced = np.array(new_y)
    return X_train_sliced, y_train_sliced

# Initialize data and paradigm
dataset = BNCI2014_001()
paradigm = MotorImagery(events=["left_hand","right_hand"], n_classes=2, fmin=8.0, fmax=32.0, tmin=0.5, tmax=2.5)

# Define search space
windows = np.arange(0.9, 0.1, -0.05)
steps = np.arange(0.03, 0.6, 0.03)
aug_parameters = [(w, s) for w in windows for s in steps if s < w and s + w < 1]

# Get reference for N_list
X_ref, y_ref, metadata_ref = paradigm.get_data(dataset=dataset, subjects=[1])
train_mask_ref = metadata_ref['session'] == '0train'
X_train_all_ref = X_ref[train_mask_ref]
N_list = list(range(5, X_train_all_ref.shape[0]//2 - 15, 5))

detailed_results = []

# Load existing results if they exist (Recovery)
if os.path.exists(checkpoint_file):
    print(f"Loading existing checkpoints from {checkpoint_file}...")
    detailed_results = pd.read_csv(checkpoint_file).to_dict('records')
else:
    print(f"No existing checkpoint file found at {checkpoint_file}. Starting new results.")

# Optimisation 1: Pre-cache dataset representations in memory to avoid repeated parsing and loading
print("Caching dataset subjects into RAM...")
subject_cache = {}
for subject_id in dataset.subject_list:
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    train_mask = metadata['session'] == '0train'
    test_mask = metadata['session'] == '1test'
    left_mask = y == 'left_hand'
    right_mask = y == 'right_hand'

    subject_cache[subject_id] = {
        'X': X,
        'y': y,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'left_mask': left_mask,
        'right_mask': right_mask
    }

seed_list = list(range(10))
seed_list.append(42)

for seed in seed_list:
    for n_trials in N_list:
        print(f"Processing Seed={seed}, N={n_trials}...")
        for subject_id in dataset.subject_list:
            # Skip if already processed in checkpoint (checks seed, subject, and N)
            if any(r.get('seed') == seed and r['subject'] == subject_id and r['n_trials_per_class'] == n_trials for r in detailed_results):
                continue

            # Load cached metadata and features
            cache_data = subject_cache[subject_id]
            X, y = cache_data['X'], cache_data['y']
            train_mask, test_mask = cache_data['train_mask'], cache_data['test_mask']
            left_mask, right_mask = cache_data['left_mask'], cache_data['right_mask']

            all_indices_left = np.where(train_mask & left_mask)[0]
            all_indices_right = np.where(train_mask & right_mask)[0]

            np.random.seed(seed)
            selected_left = np.random.choice(all_indices_left, n_trials, replace=False)
            selected_right = np.random.choice(all_indices_right, n_trials, replace=False)

            trial_indices_str = ",".join(map(str, np.concatenate([selected_left, selected_right])))

            X_train_left, y_train_left = X[selected_left], y[selected_left]
            X_train_right, y_train_right = X[selected_right], y[selected_right]
            X_test, y_test = X[test_mask], y[test_mask]

            X_train_base = np.concatenate([X_train_left, X_train_right], axis=0)
            y_train_base = np.concatenate([y_train_left, y_train_right], axis=0)

            # Baseline (No Aug) - Optimisation 2: Reuse fitted CSP features for both LDA and QDA
            csp_base = CSP(n_components=4, reg=None, log=True)
            X_tr_base_csp = csp_base.fit_transform(X_train_base, y_train_base)
            X_te_base_csp = csp_base.transform(X_test)

            for clf_name, clf_obj in [('LDA', LDA()), ('QDA', QDA(reg_param=0.1))]:
                clf_obj.fit(X_tr_base_csp, y_train_base)
                acc = np.mean(clf_obj.predict(X_te_base_csp) == y_test)
                detailed_results.append({
                    'seed': int(seed),
                    'subject': int(subject_id),
                    'n_trials_per_class': int(n_trials),
                    'window': 1.0,
                    'step': 1.0,
                    'augmentation': False,
                    'classifier': clf_name,
                    'accuracy': float(acc),
                    'train_trial_indices': trial_indices_str
                })

            # With Augmentation
            for window, step in aug_parameters:
                w_px, s_px = round(float(window)*X.shape[2]), round(float(step)*X.shape[2])
                X_tr_l, y_tr_l = slice_trials(X_train_left, y_train_left, w_px, s_px)
                X_tr_r, y_tr_r = slice_trials(X_train_right, y_train_right, w_px, s_px)

                X_tr_aug = np.concatenate([X_tr_l, X_tr_r])
                y_tr_aug = np.concatenate([y_tr_l, y_tr_r])

                # Optimisation 2: Fit CSP once per augmentation configuration
                csp_aug = CSP(n_components=4, reg=None, log=True)
                X_tr_aug_csp = csp_aug.fit_transform(X_tr_aug, y_tr_aug)

                # Pre-slice testing set once per configuration window/step
                sliced_tests = [slice_trials([xt], [yt], w_px, s_px)[0] for xt, yt in zip(X_test, y_test)]

                for clf_name, clf_obj in [('LDA', LDA()), ('QDA', QDA(reg_param=0.1))]:
                    clf_obj.fit(X_tr_aug_csp, y_tr_aug)

                    preds = []
                    for sliced_t_trial, yt in zip(sliced_tests, y_test):
                        features = csp_aug.transform(sliced_t_trial)
                        preds.append(majority_vote(clf_obj.predict(features)) == yt)

                    detailed_results.append({
                        'seed': int(seed),
                        'subject': int(subject_id),
                        'n_trials_per_class': int(n_trials),
                        'window': float(window),
                        'step': float(step),
                        'augmentation': True,
                        'classifier': clf_name,
                        'accuracy': float(np.mean(preds)),
                        'train_trial_indices': trial_indices_str
                    })

        # Optimisation 3: Checkpoint once per n_trials loop completion to drastically reduce network I/O write overhead
        pd.DataFrame(detailed_results).to_csv(checkpoint_file, index=False)
        print(f"-> Checkpoint saved to {checkpoint_file} at completion of Seed={seed}, N={n_trials}")

print(f"Final results saved to {checkpoint_file}")
display(pd.DataFrame(detailed_results).head())