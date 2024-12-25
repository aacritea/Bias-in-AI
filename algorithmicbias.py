import numpy as np
# Column names:  Energetic, Cuddly, Soft, Quiet, Happiness
survey = np.array([
  [1, 0, 1, 1, 1],  #     Energetic, Not Cuddly, Soft, Quiet,     Happy
  [1, 1, 1, 1, 1],  #     Energetic,     Cuddly, Soft, Quiet,     Happy
  [1, 0, 1, 0, 1],  #     Energetic, Not Cuddly, Soft, Loud,      Happy
  [0, 0, 1, 0, 0],  # Not Energetic, Not Cuddly, Soft, Loud,  Not happy
  [0, 1, 0, 1, 0],  # ...
  [0, 0, 0, 1, 0],
  [1, 1, 0, 0, 1],
  [0, 1, 0, 0, 0],
  [0, 1, 0, 1, 0],
  [0, 1, 0, 0, 0],
  [1, 0, 1, 1, 1],
  [0, 1, 1, 1, 0],
  [1, 0, 1, 0, 1],
  [0, 0, 1, 0, 0],
  [0, 1, 0, 1, 0],
  [0, 0, 0, 1, 0],
  [1, 1, 0, 0, 1],
  [0, 0, 0, 0, 0],
  [1, 0, 1, 1, 1],
  [1, 1, 1, 1, 0],
  [1, 0, 1, 0, 1],
  [1, 1, 1, 0, 1],
  [0, 0, 0, 0, 1],
  [0, 0, 0, 1, 1],
  [0, 0, 1, 1, 1],
  [0, 1, 1, 1, 1]
])
# CHANGEME -- You can put in your own survey results as well.

# First four columns are our features
features_train = survey[:,0:4]
# Last column is our label
labels_train = survey[:,4]

# Keeping four surveys as our test set
test_survey = np.array([
  [1, 1, 1, 0, 1],
  [0, 0, 0, 1, 0],
  [0, 0, 1, 0, 0],
  [1, 0, 1, 0, 1]
])

features_test = test_survey[:, 0:4]
labels_test = test_survey[:,4]


