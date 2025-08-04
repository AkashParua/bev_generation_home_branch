import numpy as np
import cv2
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import os
import argparse

# Cityscapes color palette for visualization
CITYSCAPES_PALETTE = [
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32],    # bicycle
]

def segment_image_with_segformer(image_path):
    """
    Performs semantic segmentation on an image using SegFormer B4 fine-tuned on Cityscapes.
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        tuple: (masks, superimposed_output)
            - masks: numpy array of shape (H, W) with class predictions
            - superimposed_output: PIL Image with segmentation overlay
    """
    # Load the specific pre-trained SegFormer model fine-tuned on Cityscapes
    model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    
    # Initialize processor and model
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    # Prepare inputs
    inputs = processor(images=image, return_tensors="pt")
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Resize logits to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )
    
    # Get predicted segmentation masks
    predicted_segmentation = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    
    # Create colored segmentation mask
    colored_mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    for class_id in range(len(CITYSCAPES_PALETTE)):
        mask = predicted_segmentation == class_id
        colored_mask[mask] = CITYSCAPES_PALETTE[class_id]
    
    # Convert original image to numpy array
    image_np = np.array(image)
    
    # Create superimposed output (blend original image with colored mask)
    alpha = 0.6  # Transparency factor
    superimposed = cv2.addWeighted(image_np, alpha, colored_mask, 1 - alpha, 0)
    superimposed_pil = Image.fromarray(superimposed.astype(np.uint8))
    
    return predicted_segmentation, superimposed_pil

def main(image_path=None):
    """
    Main function to load an image, perform segmentation, and save results.
    
    Args:
        image_path (str, optional): Path to the input image. If None, will use command line arguments.
    """
    # If called programmatically with an image_path
    if image_path is not None:
        input_image_path = image_path
        output_dir = '.'
        prefix = ''
    else:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Perform semantic segmentation on an image using SegFormer B4.')
        parser.add_argument('--image', '-i', type=str, default='input_image.jpg',
                            help='Path to the input image (default: input_image.jpg)')
        parser.add_argument('--output-dir', '-o', type=str, default='.',
                            help='Directory to save output files (default: current directory)')
        parser.add_argument('--prefix', '-p', type=str, default='',
                            help='Prefix for output filenames (default: none)')
        
        args = parser.parse_args()
        input_image_path = args.image
        output_dir = args.output_dir
        prefix = args.prefix
        
        # Ensure output directory exists
        if output_dir != '.' and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
    
    # Check if input image exists
    if not os.path.exists(input_image_path):
        print(f"Error: Input image '{input_image_path}' not found.")
        print("Please provide a valid image path using --image or -i argument.")
        return
    
    print(f"Processing image: {input_image_path}")
    
    try:
        # Perform segmentation
        masks, superimposed_output = segment_image_with_segformer(input_image_path)
        
        # Save the masks as a colored segmentation image (using Cityscapes palette)
        colored_mask = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.uint8)
        for class_id in range(len(CITYSCAPES_PALETTE)):
            mask = masks == class_id
            colored_mask[mask] = CITYSCAPES_PALETTE[class_id]
        
        colored_mask_image = Image.fromarray(colored_mask)
        colored_mask_output_path = os.path.join(output_dir, f"{prefix}segmentation_masks_colored.png")
        colored_mask_image.save(colored_mask_output_path)
        print(f"Colored segmentation masks saved to: {colored_mask_output_path}")
        
        # Save the raw class IDs as grayscale (for analysis)
        raw_mask_image = Image.fromarray(masks.astype(np.uint8))
        raw_mask_output_path = os.path.join(output_dir, f"{prefix}segmentation_masks_raw.png")
        raw_mask_image.save(raw_mask_output_path)
        print(f"Raw class masks saved to: {raw_mask_output_path}")
        
        # Save the superimposed output
        superimposed_output_path = os.path.join(output_dir, f"{prefix}superimposed_output.png")
        superimposed_output.save(superimposed_output_path)
        print(f"Superimposed output saved to: {superimposed_output_path}")
        
        # Save raw masks as numpy array for further processing
        masks_npy_path = os.path.join(output_dir, f"{prefix}segmentation_masks.npy")
        np.save(masks_npy_path, masks)
        print(f"Raw masks array saved to: {masks_npy_path}")
        
        print("\nSegmentation completed successfully!")
        print(f"Mask shape: {masks.shape}")
        print(f"Unique classes found: {np.unique(masks)}")
        
    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
        print("Make sure you have the required dependencies installed:")
        print("pip install torch torchvision transformers pillow opencv-python numpy")

if __name__ == "__main__":
    main()
