import torch
from collections import OrderedDict

def load_paras(filename):
    if not torch.cuda.is_available():
        return torch.load(filename, map_location=torch.device('cpu'))
    else:
        return torch.load(filename)

def change_paras_name(paras_ori):
    result = OrderedDict()
    count = 0
    for key, value in paras_ori.items():
        new_key = key.replace("model.", "")
        if new_key == "crit_mask_lm_smoothed.one_hot":
            continue
        result[new_key] = value
        count += 1
    print("change paras count: ", count)
    return result

if __name__ == '__main__':
    th_model = "records//model_state_epoch_4.th"
    th_paras = load_paras(th_model)
    paras_change_name = change_paras_name(th_paras)
    torch.save(paras_change_name, "resources//pytorch_model.bin")
