from imports import *

def create_mask_from_npz(npz_file_path):
    """Creates a binary mask from polygons stored in an NPZ file."""
    with np.load(npz_file_path) as f:
        polygon_arrays = [v for k, v in f.items() if k.startswith('arr_')]
        
    mask = np.zeros((256, 256), dtype=np.uint8)
    for polygons in polygon_arrays:
        if polygons.ndim == 2 and polygons.shape[1] == 2:
            mask = cv2.fillPoly(mask, pts=[polygons.astype(np.int32)], color=255)
    return mask.astype(np.float32)  # Return mask in original range [0, 255]

def preprocess_dataset(image_dir_A, image_dir_B, mask_dir, output_dir):
    """
    Processes all valid image pairs and corresponding labels, saving masks as both individual files (JPEG) and NPZ file.
    
    Args:
        image_dir_A (str): Directory containing images from set A.
        image_dir_B (str): Directory containing images from set B.
        mask_dir (str): Directory containing labels (masks) in NPZ format.
        output_dir (str): Directory to save processed data and masks.
    """
    images_A = []
    images_B = []
    masks = []

    # Output directories for images and masks
    os.makedirs(os.path.join(output_dir, "images_A"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images_B"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    # Process each file
    for img_name in os.listdir(image_dir_A):
        img_A_path = os.path.join(image_dir_A, img_name)
        img_B_path = os.path.join(image_dir_B, img_name)
        mask_path = os.path.join(mask_dir, os.path.splitext(img_name)[0] + '.npz')

        if os.path.exists(mask_path):  # Only process if mask exists
            try:
                # Load images
                image_A = Image.open(img_A_path).convert('RGB')
                image_B = Image.open(img_B_path).convert('RGB')

                # Resize images to 256x256 and convert to numpy arrays
                image_A = np.array(image_A.resize((256, 256)))
                image_B = np.array(image_B.resize((256, 256)))

                # Normalize images to [0, 1] range
                image_A = image_A.astype(np.float32) / 255.0
                image_B = image_B.astype(np.float32) / 255.0

                # Create mask
                mask = create_mask_from_npz(mask_path)

                # Save images and mask to output directories
                image_A_filename = os.path.join(output_dir, "images_A", os.path.splitext(img_name)[0] + ".jpg")
                image_B_filename = os.path.join(output_dir, "images_B", os.path.splitext(img_name)[0] + ".jpg")
                mask_filename_png = os.path.join(output_dir, "masks", os.path.splitext(img_name)[0] + ".png")
                mask_filename_jpg = os.path.join(output_dir, "masks", os.path.splitext(img_name)[0] + ".jpg")

                # Save the images and mask as required
                cv2.imwrite(image_A_filename, (image_A * 255).astype(np.uint8))  # Save A image
                cv2.imwrite(image_B_filename, (image_B * 255).astype(np.uint8))  # Save B image
                cv2.imwrite(mask_filename_png, mask)  # Save mask as PNG
                cv2.imwrite(mask_filename_jpg, mask)  # Save mask as JPG

                # Append data to lists for npz saving
                images_A.append(image_A)
                images_B.append(image_B)
                masks.append(mask)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    # Save processed dataset (images_A, images_B, masks) as a compressed .npz file
    np.savez_compressed(os.path.join(output_dir, "processed_data.npz"),
                        images_A=np.array(images_A),
                        images_B=np.array(images_B),
                        masks=np.array(masks))
    print(f"Processed dataset and masks saved to {output_dir}")

if __name__ == "__main__":
    import argparse

    # Set up argument parsing for command line inputs
    parser = argparse.ArgumentParser(description="Preprocess dataset for change detection.")
    parser.add_argument("--image_dir_A", type=str, required=True, help="Directory containing images from set A.")
    parser.add_argument("--image_dir_B", type=str, required=True, help="Directory containing images from set B.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing label masks in NPZ format.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save processed data and masks.")

    args = parser.parse_args()

    # Run the preprocessing function
    preprocess_dataset(args.image_dir_A, args.image_dir_B, args.mask_dir, args.output_dir)