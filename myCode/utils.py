import pytorch_lightning as pl
import torch
import copy

def loadModelCheckpoint(path, model):
    model_copy = copy.deepcopy(model)
    try:
        temp = torch.load(path)['state_dict']
        del temp['loss.weight']
        model_copy.load_state_dict(temp)
        print("Model Loaded")
    except Exception as e:
        print('Unable to load pytorch lightning, trying without loss.weight' + str(e))
        try:
            temp = torch.load(path)['state_dict']
            model_copy.load_state_dict(temp)
            print("Model Loaded")
        except Exception as e:
            print('Unable to load pytorch lightning, trying pytorch' + str(e))
            try:
                temp = torch.load(path)
                model_copy.model.load_state_dict(temp)
                print("Model Loaded")
            except Exception as e:
                print("Cannot load: ", path)
                return None
    return model_copy

