def fix_state_dict(state_dict):
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any(k.startswith("model.") for k in state_dict):
        state_dict = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }

    return state_dict
