import torch
from bayesian_model import BayesianModel

# Instance of model
model = BayesianModel()
model.load_state_dict(torch.load("train_network.pth"))
model.eval() # - shouldn't be used during run time to make sure dropout still works, but I think it should probably be used during tracing so that it gets the correct connection values


# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 256, 256)
print(model(example).argmax(1))
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("semanticCPPModel.pt")
print("Done")
