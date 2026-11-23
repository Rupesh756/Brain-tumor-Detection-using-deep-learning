
from pathlib import Path

from inference.model import load_model

checkpoint_path = (
    Path(__file__).parent
    / "checkpoints"
    / "best_model_v3.pth"
)

model, device = load_model(checkpoint_path)

print("\nModel architecture:")
print(model)

print("\nModel loading test completed successfully.")
