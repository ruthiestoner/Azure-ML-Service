
# Create a 2-part scoring script that computes an area of a circle, given the estimate within the pi_estimate model:
#
# 1.  The init method that loads the model. You can retrieve registered model using Model.get_model_path method
# 2.  The run method that gets invoked when you call the web service. It computes the area of a circle using 
#     area=π∗radius^2. The inputs and outputs are passed as json-formatted strings
import pickle, json
from azureml.core.model import Model

def init():
    global pi_estimate
    model_path = Model.get_model_path(model_name = "pi_estimate")
    with open(model_path, "rb") as f:
        pi_estimate = float(pickle.load(f))

def run(raw_data):
    try:
        radius = json.loads(raw_data)["radius"]
        result = pi_estimate * radius**2
        return json.dumps({"area": result})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})