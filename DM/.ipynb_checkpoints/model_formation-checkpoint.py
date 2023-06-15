from torch import nn

def make_model(input_size = 118, hidden_sizes  = [200, 300, 200, 118], output_size=12):
    model = nn.sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLu(),
                          nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                          nn.ReLu(),
                          nn.Linear(hidden_sizes[3], output_size),
                          nn.softmax(dim=1))
    return model