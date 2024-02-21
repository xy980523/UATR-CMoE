import os,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import normalize


name = "confusion_matrix.npy"
title = "CMoE Model"
baseline_data = np.load(name)
print(baseline_data)

SMALL_SIZE = 8
MEDIUM_SIZE = 16
BIGGER_SIZE = 20


normalized_baseline_data = normalize(baseline_data)
plt.rcParams["figure.figsize"] = (8, 6)
plt.title(title, size=MEDIUM_SIZE)
labels = ['Dredger','Fishboat','Motorboat','Musselboat','Naturalnoise','Oceanliner','Passengers','RORO','Sailboat']

plt.imshow(normalized_baseline_data, cmap='Oranges')
plt.xticks(np.arange(9), labels,rotation=45)
plt.yticks(np.arange(9), labels)
plt.xlabel('Predicted class', size=MEDIUM_SIZE)
plt.ylabel('True class', size=MEDIUM_SIZE)
plt.colorbar()

plt.savefig(name.replace('npy', 'jpg'), dpi=600, bbox_inches="tight")