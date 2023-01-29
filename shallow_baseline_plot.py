import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#extracted from baseline_training_log
by_alpha = [
    [89.54248366013073, 88.23529411764706, 89.54248366013073,84.31372549019608, 88.23529411764706],
    [90.19607843137256, 89.54248366013073, 89.54248366013073, 86.9281045751634, 88.23529411764706],
    [90.19607843137256, 90.19607843137256, 89.54248366013073, 88.23529411764706, 88.88888888888889],
    [90.84967320261438, 90.19607843137256, 90.19607843137256, 88.23529411764706, 88.88888888888889],
    [90.84967320261438, 90.84967320261438, 90.19607843137256, 88.23529411764706, 88.23529411764706],
    [90.84967320261438, 90.84967320261438, 90.19607843137256, 88.88888888888889, 89.54248366013073]
]

figure = plt.figure(figsize=(10, 5), layout='constrained')
p1 = plt.plot([128, 256, 512, 600, 700], by_alpha[0], label = 'alpha = 4')
p2 = plt.plot([128, 256, 512, 600, 700], by_alpha[1], label = 'alpha = 5')
p3 = plt.plot([128, 256, 512, 600, 700], by_alpha[2], label = 'alpha = 6')
p4 = plt.plot([128, 256, 512, 600, 700], by_alpha[3], label = 'alpha = 7')
p5 = plt.plot([128, 256, 512, 600, 700], by_alpha[4], label = 'alpha = 8')
p6 = plt.plot([128, 256, 512, 600, 700], by_alpha[5], label = 'alpha = 9')
plt.xticks([128,256,512,600,700])
plt.legend()
plt.title('Shallow baseline model validation accuracy vs hidden layer size for different regularization parameters')
plt.show()
