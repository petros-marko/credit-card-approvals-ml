import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("credit_card_approval_dataset.csv")

gender = data['Gender']
approved = data['Approved']

figure, axis = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')

axis[0].bar(['Approved', 'Denied'], [sum(approved), len(approved) - sum(approved)])
axis[0].set_title('Approval Status Breakdown')

axis[1].bar(['Male', 'Female'], [sum(gender), len(gender) - sum(gender)])
axis[1].set_title('Gender Breakdown')

width = 0.35
mapprovals = [ap for ap, g in zip(list(approved), list(gender)) if g == 1]
wapprovals = [ap for ap, g in zip(list(approved), list(gender)) if g == 0]
axis[2].set_xticks(np.arange(2) + width / 2)
axis[2].set_xticklabels(['Approved', 'Denied'])
m = axis[2].bar(np.arange(2), [sum(mapprovals), len(mapprovals) - sum(mapprovals)], width, color = 'r')
w = axis[2].bar(np.arange(2) + width, [sum(wapprovals), len(wapprovals) - sum(wapprovals)], width, color = 'b')
axis[2].legend((m, w), ('Male', 'Female'))
axis[2].set_title('Appoval Status by Gender')
plt.show()
