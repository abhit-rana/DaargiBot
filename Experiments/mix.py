import markov
import torch

# Create a new project for this model training experiment
# my_project = markov.Project(name="My first project try2")
my_project = markov.Project.from_id("3M35znanqpktvp")

# Define model here
hyper_parameters = {"learning_rate":0.1, "n_input":100}
model = torch.nn.Sequential()

# Use the ExperimentRecorder constuctor provided by the MarkovML SDK
# to create a new experiment recorder
recorder = markov.ExperimentRecorder(
    # Name of the experiment recording
    name="Test_Experiment_Tracking_MarkovML",
    # Project associated with the experiment
    project_id=my_project.project_id,
    # Hyper-parameters used for model training
    hyper_parameters={
        "learning_rate": 0.1,
        "n_input": 100,
        "n_hidden": 50,
        "n_output": 10
    },
    # Additional notes (optional)
    notes="This is a test experiment"
)