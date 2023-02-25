import markov
from markov.api.model.recording import ModelRecorder
from markov.api.schemas.model_recording import ModelRecordingConfig as mrc
from markov.api.schemas.model_recording import SingleTagInferenceRecord as stir


# this is that you set the metric for your model to be evaluated upon and then you just add up the records and then the model calculated the value of the metric you listed upon the rows you just added


recorder1 = ModelRecorder(mrc(
    name='model1',
    user_data_id = 'tweets_dataset',
    user_model_id = 'testing_the_feature_logisticRegression',
    model_name='Logistic_Regression',
    model_class = 'tagging',
    note='Test twitter dataset',
    info={"schema":["prediction", "actual"]})
)

recorder1.register()

for i in range(len(log_clf_pred)):
    recorder1.add_record(stir(urid=str(i), inferred=log_clf_pred[i], actual=y_test[i]))

recorder1.finish()