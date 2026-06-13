### IMPORTS

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode

### PHASE ONE

dataset = BNCI2014_001()
paradigm = MotorImagery(
    events=["left_hand","right_hand"],
    n_classes=2,           # Binary simplification (Left vs. Right)
    fmin=8.0, fmax=32.0,   # Bandpass filter (Alpha/Beta bands)
    tmin=0.5, tmax=2.5     # Epoching (0.5s to 2.5s post-cue)
)

dataset.subject_list = [1]

# X = The EEG signals (Trials x Channels x Time)
# y = The labels ('left_hand' or 'right_hand')
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

# Step 0.1: Separate training and testing data
train_mask = metadata['session'] == '0train'
test_mask  = metadata['session'] == '1test'

X_train_all = X[train_mask]
y_train_all = y[train_mask]

# Step 1: define the increments
N = list(range(50, X_train_all.shape[0] - 30, 10))

# Step 2: Prepare storage
lda_accuracies = []
qda_accuracies = []

#Step 3: Start a loop
for i in N:
    # Step 4: Slice the data
    X_train = X_train_all[:i]
    y_train = y_train_all[:i]
    X_test = X[test_mask]
    y_test = y[test_mask]

    # Step 5: Constructing the LDA pipeline
    lda_pipeline = Pipeline([
        ('csp', CSP(n_components=4, reg=None, log=True)),
        ('lda', LDA())
    ])

    # Step 6: Constructing the QDA pipeline
    qda_pipeline = Pipeline([
        ('csp', CSP(n_components=4, reg=None, log=True)),
        ('qda', QDA(reg_param=0.1))
    ])

    lda_pipeline.fit(X_train, y_train)   # CSP learns filters, LDA learns boundaries
    lda_acc = lda_pipeline.score(X_test, y_test)  # predict on test, compare to y_test
    lda_accuracies.append(lda_acc)

    qda_pipeline.fit(X_train, y_train) # See above for LDA, but for QDA now
    qda_acc = qda_pipeline.score(X_test, y_test)
    qda_accuracies.append(qda_acc)

plt.figure(figsize=(10, 6))

plt.plot(N, lda_accuracies, color='blue', marker='o', label='LDA')
plt.plot(N, qda_accuracies, color='red',  marker='o', label='QDA')

plt.axhline(y=0.5, color='gray', linestyle='--', label='Chance level (50%)') #Any result below this line means the model is actively doing worse than random, which is a red flag.

plt.xlabel('Number of Training Trials (N)')
plt.ylabel('Test Accuracy')
plt.title('LDA vs QDA: Effect of Training Set Size')
plt.legend()
plt.grid(True)
plt.show()

### PHASE TWO

# Slicing the data:
# write the function
def slice_trial(trial_data, window_size, step_size):
    #initialize empty list
    chunks = []

    #initial start_index and end_index, created outside loop and updated inside it
    start_index = 0
    end_index = start_index + window_size

    #define data size
    total_length = trial_data.shape[1]  # number of timepoints

    #create the loop
    while end_index <= total_length:

        #slice the data
        chunk = trial_data[:, start_index:end_index]
        chunks.append(chunk)

        #update the indexes
        start_index += step_size
        end_index = start_index + window_size
    #return the new data
    return np.array(chunks)

# Majority vote
def majority_vote(predictions):
    result = mode(predictions, keepdims=False) # AI said keepdims is just to avoid headaches between different SciPy versions since it returns a plain scalar
    return int(result.mode)

### PHASE THREE
def expand_training_data(X_train, y_train, window_size, step_size):
    #Step 1: Create new empty lists
    new_X = []
    new_y = []

    #Step 2: Loop through the training data and slice each one and tag each window with that trial's original label
    for trial, label in zip(X_train, y_train):
        trial_chunks = slice_trial(trial, window_size, step_size)  # (n_chunks, channels, window_size)
        new_X.append(trial_chunks)
        new_y.extend([label] * trial_chunks.shape[0])  # repeat the label once per chunk

    #flatten the per-trial chunk arrays into one big matrix
    X_train_sliced = np.vstack(new_X)

    #convert to a numpy array
    y_train_sliced = np.array(new_y)

    return X_train_sliced, y_train_sliced