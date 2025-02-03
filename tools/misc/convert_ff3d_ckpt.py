import os

import torch


def modify_state_dict_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("imgpts_neck"):
            new_key = key.replace("imgpts_neck", "pts_fusion_layer", 1)
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


# Example usage
if __name__ == "__main__":
    orig_folder = "ckpts/focalformer3d/"
    output_folder = "ckpts/focalformer3d_converted/"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(orig_folder):
        if filename.endswith(".pth"):
            orig_path = os.path.join(orig_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load the checkpoint
            checkpoint = torch.load(orig_path)

            # Modify the state_dict keys
            if "state_dict" in checkpoint:
                checkpoint["state_dict"] = modify_state_dict_keys(checkpoint["state_dict"])
            else:
                checkpoint = modify_state_dict_keys(checkpoint)

            # Save the modified checkpoint
            torch.save(checkpoint, output_path)
