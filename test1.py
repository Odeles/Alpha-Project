from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

dataset = BNCI2014_001()
paradigm = MotorImagery(
    n_classes=2,           # Binary simplification (Left vs. Right)
    fmin=8.0, fmax=32.0,   # Bandpass filter (Alpha/Beta bands)
    tmin=0.5, tmax=2.5     # Epoching (0.5s to 2.5s post-cue)
)

dataset.subject_list = [1]

# X = The EEG signals (Trials x Channels x Time)
# y = The labels ('left_hand' or 'right_hand')
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

print(f"Data shape: {X.shape}") # Expected: (~144 trials, 22 channels, 500 timepoints)