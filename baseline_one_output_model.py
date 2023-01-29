import numpy as np
from data_loader import load_data_for_approval_only 
from sklearn.neural_network import MLPClassifier

def accuracy(Y, Yhat):
    """
    Function for computing accuracy
    
    Y is a vector of the true labels and Yhat is a vector of estimated 0/1 labels
    """

    acc = 0
    for y, yhat in zip(Y, Yhat):

        if y == yhat: acc += 1

    return acc/len(Y) * 100

Xmat_train, Xmat_val, Xmat_test, Y_train, Y_val, Y_test = load_data_for_approval_only()

n, d = Xmat_train.shape

logistic_regression = MLPClassifier(solver="adam", max_iter = 2000, alpha = 0, hidden_layer_sizes=tuple(), random_state = 42)
shallow_and_wide = [MLPClassifier(solver="adam", max_iter=5000,learning_rate_init=0.01, alpha = a, hidden_layer_sizes=(w,), random_state=30) for a in range(4,10) for w in [128,256,512,600,700]]
deep_and_narrow  = [MLPClassifier(solver="adam", max_iter=5000,learning_rate_init=0.01, alpha = a, hidden_layer_sizes=(w1,w2,128), random_state=30) for a in range(4,10) for w1 in [256,512,600] for w2 in [256, 512]]

models = [logistic_regression] #+ shallow_and_wide + deep_and_narrow

for i, model in enumerate(models):
    print("Fitting model", i + 1, "/", len(models))
    print(model)
    model.fit(Xmat_train, Y_train)
    print("Training accuracy:", accuracy(Y_train, model.predict(Xmat_train)))
    print("Validation accuracy:", accuracy(Y_val, model.predict(Xmat_val)))

baseline_model = max(models, key=lambda m: accuracy(Y_val, m.predict(Xmat_val)))
print("Validation accuracy of best model:", accuracy(Y_val, baseline_model.predict(Xmat_val)))
print(baseline_model)

male = Xmat_train[0][0]
female = Xmat_train[3][0]

def probTrue(z):

    predictions = baseline_model.predict(Xmat_test)
    genders = Xmat_test.transpose()[0]

    #number of samples where gender = z
    Zs = [i for i in range(len(genders)) if genders[i] == z]
    Nz = len(Zs)

    #positive rate when gender = z
    TPzPlusFPz = sum([predictions[i] for i in Zs])

    return (TPzPlusFPz) / Nz

def probCorrect(y, z):

    predictions = [abs(1 - y - p) for p in baseline_model.predict(Xmat_test)]
    genders = Xmat_test.transpose()[0]
    Zs = [i for i in range(len(genders)) if genders[i] == z]
    Ys = [predictions[i] for i in Zs if Y_test[i] == y]

    TPz = sum(Ys)
    TPzPlusFNz = len(Ys)
    return TPz/(TPzPlusFNz)

def parityGap(male, female):
    return abs(probTrue(male) - probTrue(female))

def equalityGap(y, male, female):
    return abs(probCorrect(y, male) - probCorrect(y, female))

print(parityGap(male, female))
print(equalityGap(0, male, female))
print(equalityGap(1, male, female))
print("Test accuracy of best model:", accuracy(Y_test, baseline_model.predict(Xmat_test)))
