import numpy as np
from data_loader import load_data_for_approval_and_gender
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
from tensorflow.keras.regularizers import L2

def accuracy(Y, Yhat):
    """
    Function for computing accuracy
    
    Y is a vector of the true labels and Yhat is a vector of estimated 0/1 labels
    """

    acc = 0
    for y, yhat in zip(Y, Yhat):

        if y == yhat: acc += 1

    return acc/len(Y) * 100

Xmat_train, Xmat_val, Xmat_test, Y_approved_train, Y_approved_val, Y_approved_test, Y_gender_train, Y_gender_val, Y_gender_test = load_data_for_approval_and_gender()

n, d = Xmat_train.shape

def createDualOutputModel(input_size, hidden_layer_sizes, alpha):
        layers = [Input(shape=(input_size,), name='input')]
        for width in hidden_layer_sizes:
            layers.append(Dense(width, activation='relu', kernel_regularizer=L2(alpha))(layers[-1]))
        output_1 = Dense(1, activation='sigmoid', name='output_1', kernel_regularizer=L2(alpha))(layers[-1])
        output_2 = Dense(1, activation='sigmoid', name='output_2', kernel_regularizer=L2(alpha))(layers[-1])
        return Model(inputs=layers[0], outputs=[output_1,output_2])

def parseDualPredictions(predictions):
    out_1_preds = [ 1 if p >= 0.5 else 0 for p in predictions[0]]
    out_2_preds = [ 1 if p >= 0.5 else 0 for p in predictions[1]]
    return out_1_preds, out_2_preds

no_hidden = [createDualOutputModel(d, tuple(), 0)]
shallow_and_wide = [ createDualOutputModel(d,(w,), a) for a in range(3) for w in [128, 256, 512, 600, 700]]
deep_and_narrow = [createDualOutputModel(d, (w1, w2, 128), a) for a in range(3) for w1 in [256, 512, 600] for w2 in [256, 512]]

models = no_hidden + shallow_and_wide + deep_and_narrow

for model in models:
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(Xmat_train, {'output_1' : Y_approved_train, 'output_2': Y_gender_train}, epochs=1500, verbose = False)
    approval_predictions, gender_predictions = parseDualPredictions(model.predict(Xmat_val))
    print(model.summary())
    print(accuracy(Y_approved_val, approval_predictions), accuracy(Y_gender_val, gender_predictions))

def probTrue(z, Xmat, Y, model):

    approval_predictions, _ = parseDualPredictions(model.predict(Xmat))
    genders = Y

    #number of samples where gender = z
    Zs = [i for i in range(len(genders)) if genders[i] == z]
    Nz = len(Zs)

    #positive rate when gender = z
    TPzPlusFPz = sum([approval_predictions[i] for i in Zs])

    return (TPzPlusFPz) / Nz

def probCorrect(y, z, Xmat, Y, model):

    approval_predictions, _ = parseDualPredictions(model.predict(Xmat))
    predictions = [abs(1 - y - p) for p in approval_predictions]
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

male = Y_gender_train[0]
female = Y_gender_train[1]

#def filterModels(model):
#    approval_predictions, _ = parseDualPredictions(model.predict(Xmat_val))
#    return accuracy(Y_approved_val, approval_predictions) >= 82

def evaluateModel(model):
    approval_predictions, gender_predictions = parseDualPredictions(model.predict(Xmat_val))
    #return accuracy(Y_approved_val, approval_predictions) - abs(50 - accuracy(Y_gender_val, gender_predictions))
    return accuracy(Y_approved_val, approval_predictions) - parityGap(male, female, Xmat_val, Y_gender_val, model)*100 - .5*equalityGap(0, male, female, Xmat_val, Y_gender_val, model)*100 - .5*equalityGap(1, male, female, Xmat_val, Y_gender_val, model)*100
    #return parityGap(male, female, Xmat_val, Y_gender_val, model) + equalityGap(0, male, female, Xmat_val, Y_gender_val, model) + equalityGap(1, male, female, Xmat_val, Y_gender_val, model)

best_model = max(models, key = evaluateModel)
approval_predictions, gender_predictions = parseDualPredictions(best_model.predict(Xmat_val))

print("Validation accuracy of best model:", accuracy(Y_approved_val, approval_predictions))
print(best_model.summary())


print(parityGap(male, female, Xmat_test, Y_gender_test, best_model))
print(equalityGap(0, male, female, Xmat_test, Y_gender_test, best_model))
print(equalityGap(1, male, female, Xmat_test, Y_gender_test, best_model))
approval_predictions, gender_predictions = parseDualPredictions(best_model.predict(Xmat_test))
print("Test accuracy of best model:", accuracy(Y_approved_test, approval_predictions))
