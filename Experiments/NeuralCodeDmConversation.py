import pickle
import numpy as np
import pandas as pd
import markov

import random
import torch
from torch import nn
import torch.nn.functional as F
# from torchvision import datasets, transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# with open('/home/jovyan/ChatBot/data/train.pk','rb') as f:
#     train_data = pickle.load(f)

all_diseases = []
all_symptoms = []

# with open("/home/jovyan/ChatBot/data/disease.txt") as file:
#     for item in file:
#         all_diseases.append(item.split("\n")[0])


# with open("/home/jovyan/ChatBot/data/symptom.txt") as file:
#     for item in file:
#         all_symptoms.append(item.split("\n")[0])


with open("../data/disease2.txt") as file:
    for item in file:
        all_diseases.append(item.split("\n")[0])


with open("../data/symptom2.txt") as file:
    for item in file:
        all_symptoms.append(item.split("\n")[0])

# FOR THE NON CONVERSATION TYPE OF DATA
def get_all_X_y(df):
    X = []
    y = []
    for i in range(df.shape[0]):
        X.append(np.array(list(df.iloc[i, 0:132])))
        y.append(all_diseases.index(df.iloc[i, 132]))
    return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)

# FOR THE CONVERSATION TYPE OF DATA
# def get_all_X_y(data):
#     X = []
#     y = []
#     for conversation in data:
#         sample = [0 for i in range(len(all_symptoms))]
#         for symptom in list(conversation['implicit_inform_slots'].keys()):
#             if(conversation['implicit_inform_slots'][symptom]==True):
#                 sample[all_symptoms.index(symptom)] = 1
#         for symptom in list(conversation['explicit_inform_slots'].keys()):
#             if(conversation['explicit_inform_slots'][symptom]==True):
#                 sample[all_symptoms.index(symptom)] = 1
#         X.append(np.array(sample))
#         y.append(all_diseases.index(conversation['disease_tag']))
#     return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)


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
    global recorder
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
    loader = makeDataLoader(X, y)
    recorder.start()
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
        recorder.add_record({"loss":running_loss/len(loader)})
        print(f"Training loss: {running_loss/len(loader)}")
    recorder.stop()
    return model

# FOR THE CONVERSATION DATA
# X_train, y_train = get_all_X_y(train_data)

# FOR THE CSV TYPE OF DATA
train_df = pd.read_csv("../data/training.csv")
X_train, y_train = get_all_X_y(train_df)

# X_train_implicit, explicit_questions_train = get_X_splitted(train_data)

classification_model = form_nn_model_classification(132, [78, 60], 41)

project = markov.Project.from_id("3M35znanqpktvp")

recorder = markov.ExperimentRecorder(
    # Name of the experiment recording
    name="Neural Network Model Training. r:{}".format(random.randint(0, 100)),
    # Project associated with the experiment
    project_id=project.project_id,
    # Hyper-parameters used for model training
    hyper_parameters={
        "n_epochs":300,
        "learning_rate": 0.003,
        "input_neurons": 132,
        "hidden_layer_neurons": [88, 65],
        "output_layer_neurons": 41
    },
    # Additional notes (optional)
    # notes="This is a test experiment"
)

#recorder.start()
classification_model = train_model_classification(classification_model, 300, X_train, y_train, 0)
#recorder.stop()

# question_extraction_model = form_nn_model_questions(118, [78, 197, 78], 118)

# question_extraction_model = train_model_classification(question_extraction_model, 10, X_train_implicit, explicit_questions_train, 1)

def make_classification_predictions(model, X):
    prob_predictions = model(X[:])
    y_preds = []
    for probs in prob_predictions:
        y_preds.append(int(torch.argmax(probs)))
    return y_preds

# def make_questions_extraction_predictions(model, X):
#     predictions = model(X[:])
#     for i in range(len(predictions)):
#         predictions[i] = torch.tensor(list(map(lambda x: 1 if x>0.5 else 0, predictions[i].tolist())))
#     return predictions

def calculate_accuracy(y_actual, y_pred):
    count = 0
    for i in range(len(y_actual)):
        if(int(y_actual[i]) == y_pred[i]):
            count += 1
    return count/len(y_actual)

# def calculate_accuracy_question_extraction(y_actual, y_pred):
#     count = 0
#     for i in range(len(y_actual)):
#         if(y_actual[i] == y_pred[i]):
#             count += 1
#     return count/len(y_actual)

# with open('/home/jovyan/ChatBot/data/dev.pk','rb') as f:
#     valid_data = pickle.load(f)

# X_valid, y_valid = get_all_X_y(valid_data)

testing_df = pd.read_csv("../data/testing.csv")
X_valid, y_valid = get_all_X_y(testing_df)

y_valid_predictions = make_classification_predictions(classification_model, X_valid)

valid_accuracy = calculate_accuracy(y_valid, y_valid_predictions)
print(valid_accuracy)

# X_valid_implicit, explicit_questions_valid = get_X_splitted(valid_data)

# explicit_questions_valid_predictions = make_questions_extraction_predictions(question_extraction_model, X_valid_implicit)

# valid_accuracy = calculate_accuracy_question_extraction(explicit_questions_valid.tolist(), explicit_questions_valid_predictions.tolist())
# print(valid_accuracy)

# torch.save(classification_model, '/home/jovyan/ChatBot/models/classification_model')

# torch.save(question_extraction_model, '/home/jovyan/ChatBot/models/question_extraction_model')
