
import torch
import numpy as np

from monai.inferers import sliding_window_inference


def predict_segmentation(model, image_tensor, device):
    """
    Generate a 3D tumour segmentation mask.

    Input:
        image_tensor: (4, H, W, D)

    Output:
        prediction: (H, W, D)
        Values: 0, 1, 2, 3
    """

    model.eval()

    # Add batch dimension: (1, 4, H, W, D).
    inputs = image_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():

        logits = sliding_window_inference(
            inputs=inputs,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
            sw_device=device,
            device=device
        )

        # Convert output probabilities/logits into class labels.
        prediction = torch.argmax(logits, dim=1)

    prediction = prediction[0].cpu().numpy().astype(np.uint8)

    return prediction
