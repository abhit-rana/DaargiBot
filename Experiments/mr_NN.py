import pandas as pd
import markov

import os
import uuid

from markov.api.schemas.model_recording import SingleTagInferenceRecord

er = markov.ExperimentRecorder.from_id("hp-ArZLzLJgYQbuAv4EzpaxRTp")
my_model = markov.Model.from_id("6XFyfQEiYB8fYU3atv")

# CREATING THE EVALUATION RECORDER
evaluation_recorder = markov.EvaluationRecorder(
    name=f"Evaluating model {my_model.name}",
    notes=f"Testing evaluation with MarkovML",
    model_id=my_model.model_id
)

with open("nn_disease_classification.csv", "r") as f:
    for line in f:
        # Assign a unique identifier for individual records
        record_id = str(uuid.uuid4())
        tokens = line.strip('\n').split(',')
        record = SingleTagInferenceRecord(
            urid=record_id,
            inferred=tokens[0],
            actual=tokens[1],
            score=float(tokens[2])
        )
        evaluation_recorder.add_record(record)

evaluation_recorder.finish()