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

from sklearn.neural_network import MLPClassifier
from warnings import filterwarnings
filterwarnings('ignore')

# Define the model
mlp = MLPClassifier(hidden_layer_sizes=(4,),  # CHANGEME
                    activation='tanh',        # ADVANCED_CHANGEME
                    max_iter=1000,            # CHANGEME
                    random_state=1   
                   )

# Train the model
mlp.fit(features_train, labels_train)

print("Training set score: %f" % mlp.score(features_train, labels_train))
print("Testing set score: %f" % mlp.score(features_test, labels_test))

# Energetic, Cuddly, Soft, Quiet
features = [[0, 1, 1, 1]]    # CHANGEME
print("Yes!" if mlp.predict(features)[0] else "No!")

# Energetic, Cuddly, Soft, Quiet
features = [[1, 0, 0, 0]]    # CHANGEME
print("Yes!" if mlp.predict(features)[0] else "No!")

# Split the survey up into the cat and dog entries
dog_survey = survey[:-4]  
cat_survey = survey[-4:]

# Import plotting library
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ind = np.arange(1, 4)

# Add up the number of survey participants who are happy and
# divide by the total number of participants of each type
happy_dog = 100*np.sum(dog_survey,axis=0)[-1]/dog_survey.shape[0]
happy_cat = 100*np.sum(cat_survey,axis=0)[-1]/cat_survey.shape[0]
happy = 100*np.sum(survey, axis=0)[-1]/survey.shape[0]

# Make a bar chart
pt, pd, pc = plt.bar(ind, (happy, happy_dog, happy_cat))

# Assign colors to bars
pt.set_facecolor('b')
pd.set_facecolor('r')
pc.set_facecolor('g')

# Put labels on everything
ax.set_xticks(ind)
ax.set_xticklabels(['Happy', 'Happy | Dog', 'Happy | Cat'])
ax.set_ylim([0, 100])
ax.set_ylabel('Percent')
_ = ax.set_title('Which Pet?')

# Import library to make plots
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ind = np.arange(1, 3)

# Count the number of responses from dog vs cat owners
dog = dog_survey.shape[0]
cat = cat_survey.shape[0]

# Make a bar chart
pd, pc = plt.bar(ind, (dog, cat))

# Assign colors to bars
pd.set_facecolor('r')
pc.set_facecolor('g')

# Put labels on everything
ax.set_xticks(ind)
ax.set_xticklabels(['# Dog', '# Cat'])
ax.set_ylim([0, 25])
ax.set_ylabel('Number')
_ = ax.set_title('Which Pet?')

# Import libraries to build a plot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
fig, ax = plt.subplots(1, 1, figsize=(10,5))
ind = np.arange(0, 4)

# Count number of dog and cat surveys
total_dog = dog_survey.shape[0]
total_cat = cat_survey.shape[0]

# Count how often each feature is true divided by how many dogs and cats we have
cat_probabilities = 100*cat_survey[:,:4].sum(axis=0)/total_cat
dog_probabilities = 100*dog_survey[:,:4].sum(axis=0)/total_dog

# Input the data into a bar plot
data = {'Feature':[], 'Animal':[], 'Probability':[]}
for feature in range(4):
  data['Feature'].append(feature)
  data['Animal'].append('dog')
  data['Probability'].append(dog_probabilities[feature])

  data['Feature'].append(feature)
  data['Animal'].append('cat')
  data['Probability'].append(cat_probabilities[feature])
df = pd.DataFrame(data=data)

_ = sns.barplot(x='Feature', y='Probability', hue='Animal', data=df, ax=ax)

# Label everything
ax.set_xticklabels(['Energetic', 'Cuddly', 'Soft', 'Quiet'])
ax.tick_params(axis = 'both', which = 'major', labelsize = 24)
_ = fig.suptitle('How often is each pet ____?', fontsize=20)
_ = plt.ylabel('Probability', fontsize=18)
_ = ax.set_ylim([0, 100])
_ = plt.xlabel('Features', fontsize=18)
_ = plt.legend(loc='best', prop={'size':18})

# Import libraries for a plot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
fig, ax = plt.subplots(1, 1, figsize=(10,5))
ind = np.arange(0, 4)

# Count how often each animal is relaxed or energetic
energetic = [0,0]
energetic_count = 0
relaxed = [0,0]
relaxed_count = 0

for entry in survey:
  if entry[0] == 0:
    relaxed[entry[-1]] += 1
    relaxed_count += 1
  else:
    energetic[entry[-1]] += 1
    energetic_count += 1

# Put the values in a a database
data = {'Feature':[], 'Happy':[], 'Probability':[]}
data["Feature"].append("Energetic")
data["Happy"].append("No")
data["Probability"].append(100*energetic[0]/energetic_count)

data["Feature"].append("Energetic")
data["Happy"].append("Yes")
data["Probability"].append(100*energetic[1]/energetic_count)

data["Feature"].append("walk")
data["Happy"].append("No")
data["Probability"].append(100*relaxed[0]/relaxed_count)

data["Feature"].append("walk")
data["Happy"].append("Yes")
data["Probability"].append(100*relaxed[1]/relaxed_count)

df = pd.DataFrame(data=data)

# Plot bar plot and put labels on everything
_ = sns.barplot(x='Feature', y='Probability', hue='Happy', data=df, ax=ax)
ax.set_xticklabels(['Energetic', 'Relaxed'])
ax.tick_params(axis = 'both', which = 'major', labelsize = 24)
_ = fig.suptitle('What makes people happy?', fontsize=20)
_ = plt.ylabel('Probability', fontsize=18)
_ = ax.set_ylim([0, 100])
_ = plt.xlabel('Features', fontsize=18)
_ = plt.legend(loc='best', prop={'size':18})
