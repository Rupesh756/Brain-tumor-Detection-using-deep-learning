
from pathlib import Path

import torch
from monai.networks.nets import UNet


def build_model():
    """Create the same 3D U-Net architecture used during training."""

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2
    )

    return model


def load_model(checkpoint_path):
    """Load the trained model checkpoint."""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    device = torch.device("cpu")

    model = build_model()

    # Load the trusted checkpoint created during our training.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    saved_state = checkpoint["model_state_dict"]
    current_state = model.state_dict()

    saved_keys = list(saved_state.keys())
    current_keys = list(current_state.keys())

    # Verify that both architectures have matching parameters.
    if len(saved_keys) != len(current_keys):
        raise RuntimeError(
            "Model parameter count does not match checkpoint."
        )

    remapped_state = {}

    for old_key, new_key in zip(saved_keys, current_keys):

        if saved_state[old_key].shape != current_state[new_key].shape:
            raise RuntimeError(
                f"Parameter shape mismatch: {old_key} -> {new_key}"
            )

        remapped_state[new_key] = saved_state[old_key]

    # Strict loading ensures every model parameter matches.
    model.load_state_dict(remapped_state, strict=True)

    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    print("Device:", device)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Best validation Dice:", checkpoint.get("best_dice"))

    return model, device
