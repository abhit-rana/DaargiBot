import markov

recorder = markov.ExperimentRecorder(
    name="Hate speech classifcation using logistic regression14",
    notes="feature set consist of # of user ids, profane vector, sentiment score(compund, +ve, -ve, neutral), vecotrizer vector, and tf-idf vector",
    hyper_parameters={
        "lr":0.01,
        "epochs":1000,
        "batch_size":100,
        "w_initial": "[0. 0. .... 0.]",
        "b_initial":0,
        "iterations":100
    }
)


# example of how using to use the experiment recorder
recorder.add_record({"loss = -log loss": -l})
recorder.add_record({"weight for # of user ids feature": w[0][0]})
recorder.add_record({"sum of weights corrosponding to elements of profane vector feature": sum(w[1:210])[0]})
recorder.add_record({"weight for sentiment(compund) feature":sum(w[213:214])[0]})
recorder.add_record({"sum of weights corrosponding to elements of vectorizer vector feature": sum(w[214:16590])[0]})
recorder.add_record({"sum of weights corrosponding to elements of tfidf vector feature": sum(w[16590:32966])[0]})