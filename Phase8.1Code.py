import numpy as np
import pandas as pd
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
paradigm = MotorImagery(events=["left_hand", "right_hand"], n_classes=2, fmin=8.0, fmax=32.0, tmin=0.5, tmax=2.5)

# Define search space
windows = np.arange(0.9, 0.1, -0.05)
steps = np.arange(0.03, 0.6, 0.03)
aug_parameters = [(w, s) for w in windows for s in steps if s < w and s + w < 1]

# --- FIX (Optimisation 1 + double-load bug): cache every subject's data ONCE.
# The original script loaded subject 1 a second time just to derive N_list, and
# also assumed subject 1's trial counts were representative of every subject.
print("Caching dataset subjects into RAM...")
subject_cache = {}
min_class_trials = None
for subject_id in dataset.subject_list:
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    train_mask = metadata['session'] == '0train'
    test_mask = metadata['session'] == '1test'
    left_mask = y == 'left_hand'
    right_mask = y == 'right_hand'

    n_left = int(np.sum(train_mask & left_mask))
    n_right = int(np.sum(train_mask & right_mask))
    subject_min = min(n_left, n_right)
    min_class_trials = subject_min if min_class_trials is None else min(min_class_trials, subject_min)

    subject_cache[subject_id] = {
        'X': X,
        'y': y,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'left_mask': left_mask,
        'right_mask': right_mask,
        'n_left': n_left,
        'n_right': n_right,
    }

# FIX: N_list is now bounded by the *smallest* per-class trial count across ALL
# subjects, not just subject 1. This guarantees every (subject, n_trials) draw
# below is valid and avoids a ValueError from np.random.choice(..., replace=False).
N_list = list(range(5, min_class_trials // 2 - 15, 5))
if not N_list:
    raise ValueError(
        f"No valid N values: smallest per-class trial count across subjects is "
        f"{min_class_trials}, which is too small for the configured range."
    )

detailed_results = []
completed_keys = set()  # FIX: O(1) lookup instead of scanning detailed_results with any()

# Load existing results if they exist (Recovery)
if os.path.exists(checkpoint_file):
    print(f"Loading existing checkpoints from {checkpoint_file}...")
    existing_df = pd.read_csv(checkpoint_file)
    detailed_results = existing_df.to_dict('records')
    completed_keys = set(
        zip(existing_df['seed'], existing_df['subject'], existing_df['n_trials_per_class'])
    )
    checkpoint_file_has_header = True
else:
    print(f"No existing checkpoint file found at {checkpoint_file}. Starting new results.")
    checkpoint_file_has_header = False

seed_list = list(range(10))
seed_list.append(42)

new_rows_since_flush = []  # FIX: only these get appended to disk, instead of rewriting everything

for seed in seed_list:
    for n_trials in N_list:
        print(f"Processing Seed={seed}, N={n_trials}...")
        for subject_id in dataset.subject_list:
            # Skip if already processed in checkpoint (O(1) set lookup)
            if (seed, subject_id, n_trials) in completed_keys:
                continue

            # Load cached metadata and features
            cache_data = subject_cache[subject_id]
            X, y = cache_data['X'], cache_data['y']
            train_mask, test_mask = cache_data['train_mask'], cache_data['test_mask']
            left_mask, right_mask = cache_data['left_mask'], cache_data['right_mask']

            # Defensive guard: skip combinations that would still be invalid for
            # this specific subject (belt-and-suspenders on top of the N_list fix).
            if n_trials > cache_data['n_left'] or n_trials > cache_data['n_right']:
                print(
                    f"  Skipping subject={subject_id}, N={n_trials}: only "
                    f"{cache_data['n_left']}/{cache_data['n_right']} left/right trials available."
                )
                continue

            all_indices_left = np.where(train_mask & left_mask)[0]
            all_indices_right = np.where(train_mask & right_mask)[0]

            # FIX: use a local RNG instead of mutating global numpy random state.
            rng = np.random.default_rng(seed)
            selected_left = rng.choice(all_indices_left, n_trials, replace=False)
            selected_right = rng.choice(all_indices_right, n_trials, replace=False)

            trial_indices_str = ",".join(map(str, np.concatenate([selected_left, selected_right])))

            X_train_left, y_train_left = X[selected_left], y[selected_left]
            X_train_right, y_train_right = X[selected_right], y[selected_right]
            X_test, y_test = X[test_mask], y[test_mask]

            X_train_base = np.concatenate([X_train_left, X_train_right], axis=0)
            y_train_base = np.concatenate([y_train_left, y_train_right], axis=0)

            # Baseline (No Aug) - reuse fitted CSP features for both LDA and QDA
            # FIX: regularized covariance estimate (shrinkage) instead of reg=None,
            # which is prone to singular/ill-conditioned covariances at small N.
            csp_base = CSP(n_components=4, reg='ledoit_wolf', log=True)
            X_tr_base_csp = csp_base.fit_transform(X_train_base, y_train_base)
            X_te_base_csp = csp_base.transform(X_test)

            new_rows = []
            for clf_name, clf_obj in [('LDA', LDA()), ('QDA', QDA(reg_param=0.1))]:
                clf_obj.fit(X_tr_base_csp, y_train_base)
                acc = np.mean(clf_obj.predict(X_te_base_csp) == y_test)
                new_rows.append({
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
                w_px, s_px = round(float(window) * X.shape[2]), round(float(step) * X.shape[2])
                X_tr_l, y_tr_l = slice_trials(X_train_left, y_train_left, w_px, s_px)
                X_tr_r, y_tr_r = slice_trials(X_train_right, y_train_right, w_px, s_px)

                X_tr_aug = np.concatenate([X_tr_l, X_tr_r])
                y_tr_aug = np.concatenate([y_tr_l, y_tr_r])

                # FIX: regularized covariance here too.
                csp_aug = CSP(n_components=4, reg='ledoit_wolf', log=True)
                X_tr_aug_csp = csp_aug.fit_transform(X_tr_aug, y_tr_aug)

                # Pre-slice testing set once per configuration window/step
                sliced_tests = [slice_trials([xt], [yt], w_px, s_px)[0] for xt, yt in zip(X_test, y_test)]

                # FIX: transform each sliced test trial with csp_aug exactly ONCE
                # per (window, step) configuration, not once per classifier. The
                # transform doesn't depend on the classifier, so hoist it out of
                # the classifier loop (this halves the CSP-transform work here).
                sliced_test_features = [csp_aug.transform(sliced_t) for sliced_t in sliced_tests]

                for clf_name, clf_obj in [('LDA', LDA()), ('QDA', QDA(reg_param=0.1))]:
                    clf_obj.fit(X_tr_aug_csp, y_tr_aug)

                    preds = []
                    for features, yt in zip(sliced_test_features, y_test):
                        preds.append(majority_vote(clf_obj.predict(features)) == yt)

                    new_rows.append({
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

            detailed_results.extend(new_rows)
            new_rows_since_flush.extend(new_rows)
            completed_keys.add((seed, subject_id, n_trials))

        # FIX: append only the new rows since the last checkpoint instead of
        # rewriting the entire (ever-growing) results file from scratch.
        if new_rows_since_flush:
            pd.DataFrame(new_rows_since_flush).to_csv(
                checkpoint_file,
                mode='a' if checkpoint_file_has_header else 'w',
                header=not checkpoint_file_has_header,
                index=False,
            )
            checkpoint_file_has_header = True
            new_rows_since_flush = []
            print(f"-> Checkpoint appended to {checkpoint_file} at completion of Seed={seed}, N={n_trials}")

print(f"Final results saved to {checkpoint_file}")
display(pd.DataFrame(detailed_results).head())