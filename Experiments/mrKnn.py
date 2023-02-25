import markov
from markov import Project, Model, ModelClass
from markov import EvaluationRecorder
import pandas as pd
import random
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from markov.api.schemas.model_recording import SingleTagInferenceRecord as stir

# loading the Made Project
my_project = Project.from_id("3JYHHzor2nShd5")

# Creating the Model Manually
my_model = Model(
    name="KNN Model. r{}".format(random.randint(0, 100)),
    description="kNN Model created with project id: {}".format(my_project.project_id),
    model_class=ModelClass.TAGGING,
    project_id=my_project.project_id
)


# CREATING THE EVALUATION RECORDER
evaluation_recorder = EvaluationRecorder(
    name=f"Evaluating model: {my_model.name}",
    notes=f"Testing evaluation with MarkovML",
    model_id=my_model.model_id
)

evaluation_recorder.register()

all_diseases = []
all_diseases_sorted = ['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne', 'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma', 'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis', 'Common Cold', 'Dengue', 'Diabetes ', 'Dimorphic hemmorhoids(piles)', 'Drug Reaction', 'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia', 'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine', 'Osteoarthristis', 'Paralysis (brain hemorrhage)', 'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis', 'Typhoid', 'Urinary tract infection', 'Varicose veins', 'hepatitis A']
all_symptoms = []


with open("../data/disease2.txt") as file:
    for item in file:
        all_diseases.append(item.split("\n")[0])


with open("../data/symptom2.txt") as file:
    for item in file:
        all_symptoms.append(item.split("\n")[0])

train = pd.read_csv("../data/training.csv")
test = pd.read_csv("../data/testing.csv")

X_train = train.iloc[:,:-1]
X_test = test.iloc[:,:-1]
y_train = train.iloc[:,-1]
y_test = test.iloc[:,-1]


knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski')
knn.fit(X_train,y_train)

y_preds = knn.predict(X_test)
y_preds_probs = knn.predict_proba(X_test)


for i in range(len(y_preds_probs)):
    score = y_preds_probs[i][all_diseases_sorted.index(y_preds[i])]
    evaluation_recorder.add_record(stir(urid=str(i), inferred=all_diseases.index(y_preds[i]), actual=all_diseases.index(y_test[i]), score=score))

evaluation_recorder.finish()