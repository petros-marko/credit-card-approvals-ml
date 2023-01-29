import numpy as np
from data_loader import load_data_for_ablation
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

Xmat_train, Xmat_val, Xmat_test, Y_approved_train, Y_approved_val, Y_approved_test, Y_gender_train, Y_gender_val, Y_gender_test = load_data_for_ablation()

n, d = Xmat_train.shape

logistic_regression = MLPClassifier(solver="adam", max_iter = 2000, alpha = 0, hidden_layer_sizes=tuple(), random_state = 42)
shallow_and_wide = [MLPClassifier(solver="adam", max_iter=5000,learning_rate_init=0.01, alpha = a, hidden_layer_sizes=(w,), random_state=30) for a in range(4,10) for w in [128,256,512,600,700]]
deep_and_narrow  = [MLPClassifier(solver="adam", max_iter=5000,learning_rate_init=0.01, alpha = a, hidden_layer_sizes=(w1,w2,128), random_state=30) for a in range(4,10) for w1 in [256,512,600] for w2 in [256, 512]]

models = [logistic_regression] + shallow_and_wide + deep_and_narrow

for i, model in enumerate(models):
    print("Fitting model", i + 1, "/", len(models))
    print(model)
    model.fit(Xmat_train, Y_approved_train)
    print("Training accuracy:", accuracy(Y_approved_train, model.predict(Xmat_train)))
    print("Validation accuracy:", accuracy(Y_approved_val, model.predict(Xmat_val)))

male = Y_gender_train[0]
female = Y_gender_train[3]
print(male, female)

def probTrue(z, Xmat, Y, model):

    predictions = model.predict(Xmat)
    genders = Y

    #number of samples where gender = z
    Zs = [i for i in range(len(genders)) if genders[i] == z]
    Nz = len(Zs)

    #positive rate when gender = z
    TPzPlusFPz = sum([predictions[i] for i in Zs])

    return (TPzPlusFPz) / Nz

def probCorrect(y, z, Xmat, Y, model):

    predictions = [abs(1 - y - p) for p in model.predict(Xmat)]
    genders = Y
    Zs = [i for i in range(len(genders)) if genders[i] == z]
    Ys = [predictions[i] for i in Zs if Y_approved_test[i] == y]

    TPz = sum(Ys)
    TPzPlusFNz = len(Ys)
    return TPz/(TPzPlusFNz)

def parityGap(male, female, Xmat, Y, model):
    return abs(probTrue(male, Xmat, Y, model) - probTrue(female, Xmat, Y, model))

def equalityGap(y, male, female, Xmat, Y, model):
    return abs(probCorrect(y, male, Xmat, Y, model) - probCorrect(y, female, Xmat, Y, model))

baseline_model = max(models, key=lambda m: accuracy(Y_approved_val, m.predict(Xmat_val))) #- parityGap(male, female, Xmat_val, Y_gender_val, m)*100 - .5*equalityGap(0, male, female, Xmat_val, Y_gender_val, m)*100 - .5*equalityGap(1, male, female, Xmat_val, Y_gender_val, m)))

print("Validation accuracy of best model:", accuracy(Y_approved_val, baseline_model.predict(Xmat_val)))
print(baseline_model)


print(parityGap(male, female, Xmat_test, Y_gender_test, baseline_model))
print(equalityGap(0, male, female, Xmat_test, Y_gender_test, baseline_model))
print(equalityGap(1, male, female, Xmat_test, Y_gender_test, baseline_model))
print("Test accuracy of best model:", accuracy(Y_approved_test, baseline_model.predict(Xmat_test)))
