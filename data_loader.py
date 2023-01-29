import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('credit_card_approval_dataset.csv')

features = ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'Industry', 'Ethnicity', 'YearsEmployed', 'PriorDefault', 'Employed', 'CreditScore', 'DriversLicense', 'Citizen', 'ZipCode', 'Income', 'Approved']
#-----Do One Hot Encoding for Categorical Data------#
categorical = ['Industry', 'Ethnicity', 'Citizen']
for feature in categorical:
    col = data[feature]
    uniques = col.unique()
    onehot = pd.DataFrame({unique : [(1 if v == unique else 0) for v in col] for unique in uniques} )
    data = data.drop(columns=[feature])
    data = data.join(onehot)
#---------------------------------------------------#

#excluding zipcode because there is 170 unique values
#and so there is few repetitions and that could lead
#to problems with generalization
data = data.drop(columns=["ZipCode"])

def load_data_for_approval_only():
    Xmat = data.drop(columns=['Approved']).to_numpy()
    Y = data['Approved'].to_numpy()

    Xmat_train, Xmat_test, Y_train, Y_test = train_test_split(Xmat, Y, test_size=0.33,random_state=42)
    Xmat_train, Xmat_val, Y_train, Y_val =train_test_split(Xmat_train, Y_train, test_size=0.33,random_state=42)

    #standardize the data
    train_mean = np.mean(Xmat_train, axis=0)
    train_std = np.std(Xmat_train, axis=0)

    Xmat_train = (Xmat_train - train_mean)/train_std
    Xmat_val = (Xmat_val - train_mean)/train_std
    Xmat_test = (Xmat_test - train_mean)/train_std
    return Xmat_train, Xmat_val, Xmat_test, Y_train, Y_val, Y_test

def load_data_for_ablation():
    Xmat = data.drop(columns=['Approved', 'Gender']).to_numpy()
    Y_approved = data['Approved'].to_numpy()
    Y_gender   = data['Gender'].to_numpy()

    Xmat_train, Xmat_test, Y_approved_train, Y_approved_test, Y_gender_train, Y_gender_test = train_test_split(Xmat, Y_approved, Y_gender, test_size=0.33,random_state=42)
    Xmat_train, Xmat_val, Y_approved_train, Y_approved_val, Y_gender_train, Y_gender_val = train_test_split(Xmat_train, Y_approved_train, Y_gender_train, test_size=0.33,random_state=42)

    #standardize the data
    train_mean = np.mean(Xmat_train, axis=0)
    train_std = np.std(Xmat_train, axis=0)

    Xmat_train = (Xmat_train - train_mean)/train_std
    Xmat_val = (Xmat_val - train_mean)/train_std
    Xmat_test = (Xmat_test - train_mean)/train_std
    return Xmat_train, Xmat_val, Xmat_test, Y_approved_train, Y_approved_val, Y_approved_test, Y_gender_train, Y_gender_val, Y_gender_test

def load_data_for_approval_and_gender():
    Xmat = data.drop(columns=['Approved', 'Gender']).to_numpy()
    Y_approved = data['Approved'].to_numpy()
    Y_gender   = data['Gender'].to_numpy()

    Xmat_train, Xmat_test, Y_approved_train, Y_approved_test, Y_gender_train, Y_gender_test = train_test_split(Xmat, Y_approved, Y_gender, test_size=0.33,random_state=42)
    Xmat_train, Xmat_val, Y_approved_train, Y_approved_val, Y_gender_train, Y_gender_val = train_test_split(Xmat_train, Y_approved_train, Y_gender_train, test_size=0.33,random_state=42)

    #duplicate all rows in the training set with flipped genders
    Xmat_train_new = []
    Y_approved_train_new = []
    Y_gender_train_new = []
    for X, Ya, Yg in zip(Xmat_train, Y_approved_train, Y_gender_train):
        Xmat_train_new.append(X)
        Y_approved_train_new.append(Ya)
        Y_gender_train_new.append(Yg)
        Xmat_train_new.append(X)
        Y_approved_train_new.append(Ya)
        Y_gender_train_new.append(1 - Yg)
    Xmat_train = np.array(Xmat_train_new)
    Y_approved_train = np.array(Y_approved_train_new)
    Y_gender_train = np.array(Y_gender_train_new)

    #standardize the data
    train_mean = np.mean(Xmat_train, axis=0)
    train_std = np.std(Xmat_train, axis=0)

    Xmat_train = (Xmat_train - train_mean)/train_std
    Xmat_val = (Xmat_val - train_mean)/train_std
    Xmat_test = (Xmat_test - train_mean)/train_std
    return Xmat_train, Xmat_val, Xmat_test, Y_approved_train, Y_approved_val, Y_approved_test, Y_gender_train, Y_gender_val, Y_gender_test

load_data_for_approval_only()
load_data_for_approval_and_gender()
