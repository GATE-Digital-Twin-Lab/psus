#%%
from uqpylab import sessions
import numpy as np

from local_RP import RP14, RP22, RP111

# Start the session
mySession = sessions.cloud()
# (Optional) Get a convenient handle to the command line interface
uq = mySession.cli
# Reset the session
# mySession.reset()

uq.rng(100,'twister');

ModelOpts = { 
    'Type': 'Model', 
    'ModelFun': 'local_RP.RP14', #Not sure if this is supposed to work
    'isVectorized': 1
}
myModel = uq.createModel(ModelOpts)

# InputOpts = {
#     "Marginals": [
#         {"Name": "x1",               # Resistance
#          "Type": "Gaussian",
#          "Parameters": [0 , 1]
#         },
#         {"Name": "x2",               # Stress
#          "Type": "Gaussian",
#          "Moments": [0 , 1]
#         }
#     ]
# }

InputOpts = {
    "Marginals": [
        {"Name": "x1",               # Resistance
         "Type": "Uniform",
         "Parameters": [70 , 80]
        },
        {"Name": "x2",               # Stress
         "Type": "Gaussian",
         "Moments": [39 , 0.1]
        },
        {"Name": "x3",               # Stress
         "Type": "Gumbel",
         "Parameters": [1342, 272.9]
        },
        {"Name": "x4",               # Stress
         "Type": "Gaussian",
         "Moments": [400, 0.1]
        },
        {"Name": "x5",               # Stress
         "Type": "Gaussian",
         "Moments": [250_000, 35_000]
        },
    ]
}

myInput = uq.createInput(InputOpts)

SubsetSimOpts = {
    "Type": "Reliability",
    "Method":"Subset"     
}
SubsetSimOpts['Simulation'] = {'BatchSize': 5000}
SubsetSimAnalysis = uq.createAnalysis(SubsetSimOpts)

uq.print(SubsetSimAnalysis)

uq.display(SubsetSimAnalysis)

mySession.quit()


# %%
