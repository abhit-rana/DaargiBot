

train = pd.read_csv("./data/training.csv")
test = pd.read_csv("./data/testing.csv")

symptoms = []
disease = []
for i in range (len(train)):
    symptoms.append(train.columns[train.iloc[i]==1].to_list())
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
    with open('./data/symptom_description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            d = {row[0]: row[1]}
            description.update(d)

def getSeverity():
    global severity
    with open('./data/symptom_severity.csv') as csv_file:
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
    with open('./data/symptom_precaution.csv') as csv_file:
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

def predict_symptoms2_ask(s1, s2):
    s1 = preprocessing(s1)
    s1, ps1 = syntactic_similarity(s1, all_symptoms_preprocessed)
    if s1 == 1:
        ps1 = related_symptom(ps1)

    s2 = preprocessing(s2)
    s2,ps2 = syntactic_similarity(s2, all_symptoms_preprocessed)
    if s2 == 1:
        ps2=related_symptom(ps2)

    all_sym = [column_associated[ps1], column_associated[ps2]]
    diseases = possible_diseases(all_sym)
    # for 
    stop = False


 

def main_logic(all_symptoms):
    print("Enter main symptom: ")
    s1 = input("")
    s1 = preprocessing(s1)
    s1, ps1 = syntactic_similarity(s1, all_symptoms_preprocessed)
    if s1 == 1:
        ps1 = related_symptom(ps1)
    
    print("Enter a second symptom: ")
    s2 = input("")
    s2 = preprocessing(s2)
    s2,ps2 = syntactic_similarity(s2, all_symptoms_preprocessed)
    if s2 == 1:
        ps2=related_symptom(ps2)
    
    # the below code is basically that if there are no symptoms with the user mentioned then it tries to calculate the semantic semalirity and then ask the user whether he has the symptoms which are predicted now or not

    if s1 == 0 or s2 == 0:
        s1,ps1 = semantic_similarity(s1, all_symptoms_preprocessed)
        s2,ps2 = semantic_similarity(s2, all_symptoms_preprocessed)
        if s1 == 0:
            suggestion = suggest_syn(s1)
            for res in suggestion:
                inp=input('Do you feel '+ res+" ?(yes or no) ")
                if inp == "yes":
                    ps1 = res
                    s1 = 1
                    break
        if s2 == 0:
            suggestion = suggest_syn(s2)
            for res in suggestion:
                inp=input('Do you feel '+ res+" ?(yes or no) ")
                if inp == "yes":
                    ps2 = res
                    s2 = 1
                    break
        if s1 == 0 and s2 == 0:
            return None,None
        else:
            if s1 == 0:
                ps1 = ps2
            if s2 == 0:
                ps2 = ps1
    all_sym = [column_associated[ps1], column_associated[ps2]]
    diseases = possible_diseases(all_sym)
    stop = False
    print("Are you experiencing any ")
    for d in diseases:
        print(diseases)
        if stop == False:
            for s in symptom_disease(train, d):
                if s not in all_sym:
                    print(clean_symptoms(s)+' ?')
                    while True:
                        inp = input("")
                        if(inp == "yes" or inp == "no"):
                            break
                        else:
                            print("Provide proper answers i.e. (yes/no) : ",end="")
                    if inp == "yes":
                        all_sym.append(s)
                        diseases = possible_diseases(all_sym)
                        if len(diseases) == 1:
                            stop=True 
    return knn.predict(one_hot_vector(all_sym, all_symptoms)),all_sym


def chat():
    a = True
    while a:
        result, s = main_logic(all_symptoms)
        if result == None :
            ans3 = input("can you specify more what you feel or tap q to stop the conversation")
            if ans3 == "q":
                a = False
            else:
                continue
        else:
            print("you may have " + result[0])
            print(description[result[0]])
            an = input("how many day do you feel those symptoms ?")
            if calc_condition(s,int(an)) == 1:
                print("you should take the consultation from doctor")
            else : 
                print('Take following precautions : ')
                for e in precaution[result[0]]:
                    print(e)
            print("do you need another medical consultation (yes or no)? ")
            ans=input()
            if ans!="yes":
                a = False
                print("!!!!! thanks !!!!!! ")

chat()



'''
QUERIES:
1) WHERE IS THE DECISION TREE CLASSIFIER USED
2) KNN IS USED FOR SUGGESTION NOT FOR THE RETRIEVAL OF SYMPTOMS
3) WHY KNN IS ALSO USED SINCE IN THE MAIN LOGIC CODE WHEN THE LENGTH OF THE DIESEASES IS ONE THEN THAT IS THE ONLY DIESEASE WHICH THE PERSON WOULD BE SUFFERING

'''
