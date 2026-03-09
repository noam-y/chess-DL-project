import os
import argparse
import glob
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Run inference on a directory of unlabeled images.")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the python file implementing the model protocol (e.g., model_v3.py)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth checkpoint file")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing unlabeled images")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results. Defaults to 'results_{model_name}'")
    parser.add_argument("--ood_threshold", type=float, default=0.7, help="Confidence threshold for OOD detection")
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        model_name = os.path.splitext(os.path.basename(args.model_file))[0]
        args.output_dir = f"results_{model_name}"
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(image_files)} images. Starting inference...")

    # Path to inference_all.py (assuming it's in the same directory as this script)
    inference_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_all.py")

    for img_path in image_files:
        base_name = os.path.basename(img_path)
        output_path = os.path.join(args.output_dir, f"result_{base_name}")
        
        print(f"Processing {base_name}...")
        
        # Construct command
        cmd = [
            sys.executable, inference_script,
            "--model_file", args.model_file,
            "--checkpoint", args.checkpoint,
            "--image_path", img_path,
            "--output_path", output_path,
            "--ood_threshold", str(args.ood_threshold)
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {base_name}: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    print(f"\nAll done! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
