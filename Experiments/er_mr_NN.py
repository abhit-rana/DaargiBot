import pickle
import numpy as np
import pandas as pd
import markov

import random
import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv
import os
import uuid

# from markov.api.schemas.model_recording import SingleTagInferenceRecord
from markov.api.schemas.model_recording import SingleTagInferenceRecord as stir

all_diseases = []
all_symptoms = []


with open("../data/disease2.txt") as file:
    for item in file:
        all_diseases.append(item.split("\n")[0])


with open("../data/symptom2.txt") as file:
    for item in file:
        all_symptoms.append(item.split("\n")[0])

def get_all_X_y(df):
    X = []
    y = []
    for i in range(df.shape[0]):
        X.append(np.array(list(df.iloc[i, 0:132])))
        y.append(all_diseases.index(df.iloc[i, 132]))
    return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)



def get_X_splitted(data):
    X_implicit = []
    explicit_questions = []
    for conversation in data:
        sample_implicit = [0 for i in range(len(all_symptoms))]
        sample_explicit = [0 for i in range(len(all_symptoms))]
        for symptom in list(conversation['implicit_inform_slots'].keys()):
            if(conversation['implicit_inform_slots'][symptom]==True):
                sample_implicit[all_symptoms.index(symptom)] = 1
        for symptom in list(conversation['explicit_inform_slots'].keys()):
            sample_explicit[all_symptoms.index(symptom)] = 1
        explicit_questions.append(np.array(sample_explicit))
        X_implicit.append(np.array(sample_implicit))
    return torch.tensor(X_implicit, dtype=torch.float), torch.tensor(explicit_questions, dtype=torch.float)


def form_nn_model_classification(input_size, hidden_sizes, output_size):
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.Softmax(dim=1))
    return model


def form_nn_model_questions(input_size, hidden_sizes, output_size):
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.SiLU(),
                          nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[2], output_size),
                          nn.Sigmoid())
    return model

class ModelMyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len

def makeDataLoader(X, y):
    data = ModelMyDataset(X, y)
    loader = DataLoader(dataset=data, batch_size=64, shuffle=True)
    return loader


def train_model_classification(model, epochs, X, y, flag_4_questions):
    global er_recorder
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
    loader = makeDataLoader(X, y)
    er_recorder.start()
    for epoch in range(epochs):
        running_loss = 0
        for i, data in enumerate(loader, 0):
            x_batch, y_batch = data
            optimizer.zero_grad()
            output = model(x_batch)
            if(flag_4_questions):
                loss = loss_function(output, y_batch)
            else:
                loss = loss_function(output, torch.tensor(y_batch, dtype=torch.long))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        er_recorder.add_record({"loss":running_loss/len(loader)})
        print(f"Training loss: {running_loss/len(loader)}")
    er_recorder.stop()
    return model


train_df = pd.read_csv("../data/training.csv")
X_train, y_train = get_all_X_y(train_df)


classification_model = form_nn_model_classification(132, [78, 60], 41)

# loading the Made Project
my_project = markov.Project.from_id("3JYHHzor2nShd5")

random_no = random.randint(0, 100)

er_recorder = markov.ExperimentRecorder(
    name="Neural Network Model Training. r:{}".format(random_no),
    project_id=my_project.project_id,

    hyper_parameters={
        "n_epochs":300,
        "learning_rate": 0.003,
        "input_neurons": 132,
        "hidden_layer_neurons": [88, 65],
        "output_layer_neurons": 41
    },
)

classification_model = train_model_classification(classification_model, 300, X_train, y_train, 0)


def make_classification_predictions(model, X):
    prob_predictions = model(X[:])
    y_preds = []
    scores = []
    for probs in prob_predictions:
        y_preds.append(int(torch.argmax(probs)))
        scores.append(probs[int(torch.argmax(probs))])
    return y_preds, scores


def calculate_accuracy(y_actual, y_pred):
    count = 0
    for i in range(len(y_actual)):
        if(int(y_actual[i]) == y_pred[i]):
            count += 1
    return count/len(y_actual)


testing_df = pd.read_csv("../data/testing.csv")
X_test, y_test = get_all_X_y(testing_df)

y_test_predictions, scores = make_classification_predictions(classification_model, X_test)

# my_model = er_recorder.model_id
MODEL_NAME = "NN model. r{}".format(random_no)

evaluation_recorder = markov.EvaluationRecorder(
    name="Evaluating model:{}".format(MODEL_NAME),
    notes=f"Testing evaluation with MarkovML",
    model_id=er_recorder.model_id
)

evaluation_recorder.register()

for i in range(len(y_test_predictions)):
    evaluation_recorder.add_record(stir(urid=str(i), inferred=y_test_predictions[i], actual=int(y_test[i]), score=float(scores[i])))

evaluation_recorder.finish()