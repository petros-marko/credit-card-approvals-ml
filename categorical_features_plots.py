import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("credit_card_approval_dataset.csv")

industry = data['Industry']
ethnicity = data['Ethnicity']
citizenship = data['Citizen']

figure, axis = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')

axis[0].bar(list(industry.unique()), [ sum([1 for v in industry if v == i]) for i in industry.unique()])
axis[0].set_title('Industries')
for tick in axis[0].get_xticklabels():
    tick.set_rotation(90)

axis[1].bar(list(ethnicity.unique()), [ sum([1 for v in ethnicity if v == i]) for i in ethnicity.unique()])
axis[1].set_title('Ethnicities')

axis[2].bar(list(citizenship.unique()), [ sum([1 for v in citizenship if v == i]) for i in citizenship.unique()])
axis[2].set_title('Citizenship')

plt.show()
