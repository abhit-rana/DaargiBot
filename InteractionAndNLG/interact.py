'''
ASSUMPTIONS - VERY VERY IMPORTANT
1) CURRENTLY I AM THINKING THAT THE QUESTIONS WHICH NEED TO BE ASKED WOULD BE PREDICTED IN ONE GO AND RETURNED AS A LIST.
2) SECONDLY IF WE WANT TO ASK MORE THEN TWO SYMPTOM IN ONE TIME THEN PLEASE RETURN THEM AS AND SEPRATED.

very important way of giving reply to the symptoms asked by the bot

* every symptom must be preceded by yes or no and order of symptoms could be any (done)
* he could give yes or no with keeping in mind the indexing of the symptom asked!!! (done)
* he could write yes all, no all, or yes and no for all type of things (think)
'''


import subprocess
import json
import random
import pandas as pd
import numpy as np
import json
import csv
import itertools
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


PersonSuffering = "user"
question_asking_templates = ["was there any symptom like {}?", "was {} observed?", "Did {0} showed {1}?", "Did {0} faced {1}?"]
user_symptoms = {}
user_symptoms_yes_lst = []

def give_greetings_reply():
    return "Hey! How can I help you?"

def give_bye_reply():
    return "Bye!, take care."

def extract_symptoms_self_report(info_dict):
    global user_symptoms
    global user_symptoms_yes_lst
    entities_list = info_dict["entities"]
    global PersonSuffering
    suffering = True
    flag = False
    for entity in entities_list:
        if(entity["entity"]=="PersonSuffering"):
            PersonSuffering = entity["value"]
        elif (entity["entity"] == "suffering"):
            suffering = True
        elif (entity["entity"] == "notSuffering"):
            suffering = False
        elif (entity["entity"] == "symptom"):
            flag = True
            user_symptoms[entity["value"]] = suffering
            if(suffering):
                user_symptoms_yes_lst.append(entity['value'])
    return flag

def get_info_dict(user_reply):
    text_dict = {"title": "Wget POST","text": user_reply,"userId":1}
    process = subprocess.run(['wget', '--method=post', '-O-', '-q', '--body-data=' + json.dumps(text_dict), 'localhost:5005/model/parse'],
                            stdout=subprocess.PIPE,
                            universal_newlines=True, stderr=subprocess.PIPE, text=True)
    return eval(process.stdout)
    # text_dict = {"text": user_reply}
    # process = subprocess.run(['curl', 'localhost:5005/model/parse', '-d', json.dumps(text_dict)],
    #                         stdout=subprocess.PIPE,
    #                         universal_newlines=True, stderr=subprocess.PIPE, text=True)
    # return eval(process.stdout)



def extract_symptoms_conversation(q, info_dict):
    global user_symptoms
    global user_symptoms_yes_lst
    entities_list = info_dict["entities"]
    suffering = True
    flag = True
    questions = q.split(" and ")
    response4questions = []
    for entity in entities_list:
        if (entity["entity"] == "suffering"):
            suffering = True
            response4questions.append(suffering)
        elif (entity["entity"] == "notSuffering"):
            suffering = False
            response4questions.append(suffering)
        elif (entity["entity"] == "symptom"):
            user_symptoms[entity["value"]] = suffering
            if(suffering):
                user_symptoms_yes_lst.append(entity['value'])
            flag = False
    if(flag):
        for i in range(len(questions)):
            user_symptoms[questions[i]] = response4questions[i]
            if(suffering):
                user_symptoms_yes_lst.append(questions[i])
    diseases = possible_diseases(user_symptoms_yes_lst)
    if(len(diseases)==1):
        return True
    return False


def printQuestionAndTakeInput(questions):
    for q in questions:
        if(q=="family_history"):
            print("Is there family history with such symptoms?")
        else:
            no = random.randint(0, len(question_asking_templates)-1)
            if(no<2):
                print(question_asking_templates[no].format(clean_symptoms(q)))
            else:
                if(PersonSuffering == "user"):
                    print(question_asking_templates[no].format("you", clean_symptoms(q)))
                else:
                    print(question_asking_templates[no].format("your " + PersonSuffering, clean_symptoms(q)))
        user_reply = input(">>")
        info_dict = get_info_dict(user_reply)
        if(extract_symptoms_conversation(q, info_dict)):
            return True
    return False

def predict_questions():
    diseases = possible_diseases(user_symptoms_yes_lst)
    # print("diseases from self report:", diseases)
    stop = False
    for d in diseases:
        if stop == False:
            disease_symptoms = [symp for symp in symptom_disease(train, d) if symp not in list(user_symptoms.keys())]
            # print("disease_symptoms:", disease_symptoms)
            stop = printQuestionAndTakeInput(disease_symptoms)
        else:
            break
    return knn.predict(one_hot_vector(user_symptoms_yes_lst, all_symptoms))

def interact():
    global user_symptoms
    getDescription()
    getPrecaution()
    getSeverity()
    while(True):
        user_reply = input(">>").lower()
        info_dict = get_info_dict(user_reply)
        # print(info_dict)
        if(info_dict['intent']["name"]=='greet'):
            print(give_greetings_reply())
        elif(info_dict['intent']["name"]=='self_report'):
            if(extract_symptoms_self_report(info_dict)):
                # print("before calling the predict questions functions:", user_symptoms_yes_lst)
                disease = predict_questions()
                print()
                print("***************DIAGNOSIS RESULT********************")
                print()
                print("POSSIBLE DISEASE:", disease[0])
                print("DESCRIPTION ABOUT THE DISEASE:\n", description[disease[0]])
                an = input("FROM HOW MANY DAYS THESE SYMPTOMS ARE OBSERVED?")
                if calc_condition(user_symptoms_yes_lst,int(an)) == 1:
                    if(PermissionError!="user"):
                        print("SUGGESTION: your {} should take the consultation from doctor!!".format(PersonSuffering))    
                    else:
                        print("SUGGESTION: you should take the consultation from doctor!!")
                else : 
                    print("SUGGESTED PRECAUTIONS:")
                    for e in precaution[disease[0]]:
                        print(e)
                # print(user_symptoms)
                print()
                print("***************************************************")
            else:
                print("There is no symptom reported. Please Report atleast one symptom!!")
                continue
        elif(info_dict['intent']["name"]=='showing_gratitude'):
            print("No Problem, it's my job!")
        elif(info_dict['intent']["name"]=='goodbye'):
            print(give_bye_reply())
            break


######################################DIALOGUE MANAGER: IEEHSA######################################
train = pd.read_csv("../data/training.csv")
test = pd.read_csv("../data/testing.csv")

# symptoms = []
disease = []
for i in range (len(train)):
    # symptoms.append(train.columns[train.iloc[i]==1].to_list())
    if(train.iloc[i, -1] not in disease):
        disease.append(train.iloc[i, -1])

all_symptoms = list(train.columns[:-1]) 

def clean_symptoms(symptom):
    symptom = symptom.replace('_', ' ')
    symptom = symptom.replace('.1', '')
    symptom = symptom.replace('(typhos)', '')
    symptom = symptom.replace('yellowish', 'yellow')
    symptom = symptom.replace('yellowing', 'yellow')
    return symptom

all_symp = [clean_symptoms(symptom) for symptom in (all_symptoms)]

nlp = spacy.load('en_core_web_sm')

def preprocessing(doc):
    nlp_doc = nlp(doc)
    d = []
    for token in nlp_doc:
        if(not token.text.lower in STOP_WORDS and token.text.isalpha()):
            d.append(token.lemma_.lower())
    return ' '.join(d)

all_symptoms_preprocessed = [preprocessing(sym) for sym in all_symp]

column_associated = dict(zip(all_symptoms_preprocessed, all_symptoms))

def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item

def sort(a):
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if len(a[j]) > len(a[i]):
                a[i], a[j] = a[j], a[i]
    a.pop()
    return a

def permutations(s):
    permutations = list(itertools.permutations(s))
    return ([' '.join(permutation) for permutation in permutations])


def doesExist(text):
    text = text.split(' ')
    combinations = [x for x in powerset(text)]
    sort(combinations)
    for c in combinations:
        for symptom in permutations(c):
            if symptom in all_symptoms_preprocessed:
                return symptom
    return False

def jaccard(str1, str2):
    list1 = str1.split(' ')
    list2 = str2.split(' ')
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def syntactic_similarity(s, corpus):
    most_sim = []
    poss_sym = []
    for symptom in corpus:
        d = jaccard(s, symptom)
        most_sim.append(d)
    order = np.argsort(most_sim)[::-1].tolist()
    for i in order:
        if doesExist(s):
            return 1, [corpus[i]]
        if corpus[i] not in poss_sym and most_sim[i] != 0:
            poss_sym.append(corpus[i])
    if len(poss_sym):
        return 1, poss_sym
    else:
        return 0, None
    

def check_pattern(inp, dis_list):
    import re
    pred_list = []
    ptr = 0
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return ptr, None


def wsd(word, context):
    overalp = lesk(context, word)
    return overalp

def semantic(doc1, doc2):
    doc1_p = preprocessing(doc1).split(' ')
    doc2_p = preprocessing(doc2).split(' ')
    score = 0
    for tock1 in doc1_p:
        for tock2 in doc2_p:
            syn1 = wsd(tock1, doc1)
            syn2 = wsd(tock2, doc2)
            if syn1 is not None and syn2 is not None:
                x = syn1.wup_similarity(syn2)
                if x is not None and x > 0.25:
                    score += x
    return score / (len(doc1_p) * len(doc2_p))


def semantic_similarity(s, corpus):
    max_sim = 0
    most_sim = None
    for symptom in corpus:
        d = semantic(s, symptom)
        if d > max_sim:
            most_sim = symp
            max_sim = d
    return max_sim, most_sim


def suggest_syn(s):
    symptoms = []
    synonyms = wordnet.synsets(s)
    lemmas = [word.lemma_names() for word in synonyms]
    lemmas = list(set(itertools.chain(*lemmas)))
    for e in lemmas:
        res, s1 = semantic_similarity(e, all_symptoms_preprocessed)
        if res != 0:
            symptoms.append(s1)
    return list(set(symptoms))

def one_hot_vector(symptom, all_symptom):
    l = np.zeros([1, len(all_symptom)])
    for sym in symptom:
        l[0, all_symptom.index(sym)] = 1
    return pd.DataFrame(l, columns=all_symptoms)

def contains(small, big):
    a = True
    for i in small:
        if i not in big:
            a = False
    return a

def possible_diseases(l):
    possible_disease = []
    for d in set(disease):
        if contains(l, symptom_disease(train, d)):
            possible_disease.append(d)
    return possible_disease

def symptom_disease(df, disease):
    ddf = df[df.prognosis == disease]
    m = (ddf == 1).any()
    return m.index[m].tolist()

severity = dict()
description = dict()
precaution = dict()

def getDescription():
    global description
    with open('../data/symptom_description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            d = {row[0]: row[1]}
            description.update(d)

def getSeverity():
    global severity
    with open('../data/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                d = {row[0]: int(row[1])}
                severity.update(d)
        except:
            pass

def getPrecaution():
    global precaution
    with open('../data/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            p = {row[0]: [row[1], row[2], row[3], row[4]]}
            precaution.update(p)

def calc_condition(exp,days):
    sum=0
    for item in exp:
        if item in severity.keys():
            sum=sum+severity[item]
    if((sum*days)/(len(exp))>13):
        return 1
        print("You should take the consultation from doctor. ")
    else:
        return 0
        print("It might not be that bad but you should take precautions.")



X_train = train.iloc[:,:-1]
X_test = test.iloc[:,:-1]
y_train = train.iloc[:,-1]
y_test = test.iloc[:,-1]

knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski')
knn.fit(X_train,y_train)

dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 1)
dt.fit(X_train, y_train)


def related_symptom(s):
    if len(s) == 1:
        return s[0]
    print("Searches related to input: ")
    for n, i in enumerate(s):
        print(n, ")", clean_symptoms(i))
    if n != 0:
        print("Select the one you meant (0 - {num}): ", end = "")
        confirm_input = int(input(""))
    else:
        confirm_input = 0
    disease_input = s[confirm_input]
    return disease_input






###########################################################################################################



if __name__=="__main__":
    interact()