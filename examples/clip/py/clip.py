import numpy as np
import os
import argparse
import json
import re
from PIL import Image
from amlnnlite.api import AMLNNLite


def preprocess_image(image_path: str, target_size: int = 224) -> np.ndarray:
    """
    Preprocess image for CLIP model.
    
    Steps:
        1. Load image and convert to RGB
        2. Scale the shorter side to target_size
        3. Center crop to target_size x target_size
        4. Normalize with CLIP mean and std
    
    Args:
        image_path (str): Path to input image
        target_size (int): Target image size (default: 224)
    
    Returns:
        np.ndarray: Preprocessed image data with shape (target_size, target_size, 3)
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    
    # Scale the shorter side
    scale = target_size / min(width, height)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    
    # Resize
    img = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Center crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    img = img.crop((left, top, left + target_size, top + target_size))
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # CLIP normalization
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    
    # Normalize: (x - mean) / std
    img_array = (img_array - mean) / std
    
    # Return in NHWC format
    return img_array


def post_process(
    image_features: np.ndarray,
    text_features: np.ndarray,
    scale: float = 100.00000762939453,
    use_cosine: bool = True,
    apply_scale: bool = True,
) -> float:
    """
    Calculate similarity between image and text features.
    
    Args:
        image_features (np.ndarray): Image feature vector
        text_features (np.ndarray): Text feature vector
        scale (float): Scale factor for similarity calculation
        use_cosine (bool): If True, L2-normalize both vectors before dot product (cosine similarity)
        apply_scale (bool): If True, multiply by scale after dot product
    
    Returns:
        float: Similarity score
    """
    img_vec = image_features.flatten().astype(np.float32)
    txt_vec = np.array(text_features, dtype=np.float32).flatten()
    
    if len(img_vec) != len(txt_vec):
        raise ValueError(f"Feature dimension mismatch: image={len(img_vec)}, text={len(txt_vec)}")
    
    if use_cosine:
        img_norm = np.linalg.norm(img_vec) + 1e-8
        txt_norm = np.linalg.norm(txt_vec) + 1e-8
        img_vec = img_vec / img_norm
        txt_vec = txt_vec / txt_norm
    
    dot_product = np.dot(img_vec, txt_vec)
    
    similarity = dot_product * scale if apply_scale else dot_product
    
    return float(similarity)


def extract_index(filename: str) -> int:
    """
    Extract index from filename pattern: test_xxx_index.jpg
    
    Args:
        filename (str): Filename to extract index from
    
    Returns:
        int: Extracted index, or -1 if pattern doesn't match
    """
    pattern = r"test_\w+_(\d+)\.jpg"
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1))
    return -1


def process_image_dir(
    amlnn: AMLNNLite,
    image_dir_path: str,
    base_dir: str = "",
    json_filename: str = ""
) -> list:
    """
    Process image directory and find best matching text dataset.
    
    Args:
        amlnn: AMLNNLite instance
        image_dir_path (str): Path to directory containing test images
        base_dir (str): Base directory for clip datasets (optional, can use CLIP_BASE_DIR env var)
        json_filename (str): JSON filename in each dataset folder (optional, can use CLIP_JSON_FILENAME env var)
    
    Returns:
        list: List of best matching dataset paths
    """
    results = []
    file_pattern = re.compile(r"test_(\w+)_\d+\.jpg")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    
    if not base_dir:
        base_dir = os.getenv("CLIP_BASE_DIR", "./clip_datasets/")
    
    if not json_filename:
        json_filename = os.getenv("CLIP_JSON_FILENAME", "clip_text_res.json")
    
    matched_files = []
    if os.path.isdir(image_dir_path):
        for filename in os.listdir(image_dir_path):
            filepath = os.path.join(image_dir_path, filename)
            if os.path.isfile(filepath):
                if file_pattern.match(filename):
                    matched_files.append((filename, filepath, True))  
                elif any(filename.lower().endswith(ext) for ext in image_extensions):
                    matched_files.append((filename, filepath, False))  
    elif os.path.isfile(image_dir_path):
        filename = os.path.basename(image_dir_path)
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            has_pattern = bool(file_pattern.match(filename))
            matched_files.append((filename, image_dir_path, has_pattern))
        else:
            print(f"Error: {image_dir_path} is not a valid image file")
            return results
    else:
        print(f"Error: {image_dir_path} is not a valid directory or file")
        return results
    
    if not matched_files:
        print(f"Warning: No image files found in {image_dir_path}")
        return results
    
    print(f"Found {len(matched_files)} image file(s) to process")
    
    matched_files.sort(key=lambda x: extract_index(x[0]) if x[2] else 999999)
    
    # Process each image
    for filename, filepath, has_pattern in matched_files:
        if has_pattern:
            match = file_pattern.match(filename)
            if match:
                name = match.group(1)
            else:
                name = ""  
        else:
            name = ""
        
        # Preprocess image
        try:
            input_data = preprocess_image(filepath)
            input_data = np.expand_dims(input_data, axis=0)
        except Exception as e:
            print(f"Error preprocessing image {filename}: {e}")
            continue
        
        # Run inference
        try:
            outputs = amlnn.inference(inputs=[input_data])
            model_output = outputs[0]  
            if isinstance(model_output, np.ndarray):
                model_output = model_output.astype(np.float32)
            else:
                model_output = np.array(model_output, dtype=np.float32)
            model_output = model_output.flatten()
        except Exception as e:
            print(f"Error running inference on {filename}: {e}")
            continue
        
        max_sim = float('-inf')
        best_key = ""
        best_id = ""
        
        if not os.path.isdir(base_dir):
            print(f"Error: Base directory does not exist: {base_dir}")
            continue
        
        print(f"Searching in base directory: {base_dir}")
        folder_count = 0
        for folder_name in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            if has_pattern and name and name not in folder_name:
                continue
            
            folder_count += 1
            
            vit_res_path = os.path.join(folder_path, json_filename)
            if not os.path.isfile(vit_res_path):
                print(f"Warning: JSON file not found: {vit_res_path}")
                continue
            
            try:
                with open(vit_res_path, 'r', encoding='utf-8') as f:
                    vit_json = json.load(f)
                
                    for key, text_vec in vit_json.items():
                        if isinstance(text_vec, list):
                            text_features = np.array(text_vec, dtype=np.float32)
                            sim_scaled = post_process(
                                model_output,
                                text_features,
                                use_cosine=True,
                                apply_scale=True,
                            )
                            
                            if sim_scaled > max_sim:
                                max_sim = sim_scaled
                                best_key = key
                                best_id = folder_name
            except Exception as e:
                print(f"Error loading JSON file {vit_res_path}: {e}")
                continue
        
        if best_key and best_id:
            best_path = os.path.join(base_dir, best_id)
            results.append(best_path)
            print(f"\nProcessing image: {filename}")
            print(f"  Best matching dataset: {best_path}")
        else:
            print(f"\nProcessing image: {filename}")
            print(f"  No matching dataset found (searched {folder_count} folder(s))")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='CLIP Image-Text Matching Demo')
    parser.add_argument('--model-path', required=True, help='Path to the CLIP model file')
    parser.add_argument('--base-dir', default='./clip_datasets/', help='Base directory for clip datasets (can also use CLIP_BASE_DIR env var)')
    parser.add_argument('--json-filename', default='clip_text_res.json', help='JSON filename in each dataset folder (can also use CLIP_JSON_FILENAME env var, default: clip_text_res.json)')
    parser.add_argument('--image-dir', default='./', help='Image directory or single image file to process (optional, will prompt if not provided)')
    args = parser.parse_args()
    
    # Initialize AMLNNLite
    print("Initializing model...")
    amlnn = AMLNNLite()
    amlnn.config(model_path=args.model_path)
    amlnn.init()
    print("Model initialized successfully.\n")
    
    # Process images
    if args.image_dir:
        results = process_image_dir(amlnn, args.image_dir, args.base_dir, args.json_filename)
        print(f"\nTotal results: {len(results)}")
        for i, result in enumerate(results):
            print(f"Index[{i}]: {result}")
    else:
        while True:
            image_path = input("\nPlease enter the JPG image path or directory (enter 'exit' to quit):\n").strip()
            
            if image_path.lower() == 'exit':
                break
            
            if not image_path:
                print("The path cannot be empty.")
                continue
            
            results = process_image_dir(amlnn, image_path, args.base_dir, args.json_filename)
            
            for i, result in enumerate(results):
                print(f"Index[{i}]: {result}")
    
    amlnn.uninit()
    print("\nDone.")


if __name__ == "__main__":
    main()
