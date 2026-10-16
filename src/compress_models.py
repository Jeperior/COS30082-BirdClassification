import os
import torch

MODEL_DIR = "models"

MODEL_FILES = [
    "best_model.pth",
    "finetuned_model.pth",
    "finetuned_augmented_model.pth",
]

OUTPUT_DIR = "models_fp16"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in MODEL_FILES:
        input_path = os.path.join(MODEL_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        print(f"\nProcessing: {filename}")

        # Load the original checkpoint.
        state_dict = torch.load(
            input_path,
            map_location="cpu",
            weights_only=True
        )

        # Convert floating-point tensors to FP16.
        compressed_state_dict = {
            key: (
                value.half()
                if value.is_floating_point()
                else value
            )
            for key, value in state_dict.items()
        }

        # Save to a separate directory.
        torch.save(compressed_state_dict, output_path)

        original_size = os.path.getsize(input_path) / (1024 ** 2)
        compressed_size = os.path.getsize(output_path) / (1024 ** 2)

        print(f"Original size: {original_size:.2f} MB")
        print(f"FP16 size:    {compressed_size:.2f} MB")
        print(
            f"Reduction:    "
            f"{(1 - compressed_size / original_size) * 100:.2f}%"
        )

    print("\nConversion complete!")
    print(f"Compressed models saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()