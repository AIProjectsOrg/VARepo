import os
import cv2
import numpy as np
import random
import argparse

def load_yolo_labels(label_path, img_width, img_height):
    bboxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x_center, y_center, width, height = map(float, parts)
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            bboxes.append((x1, y1, x2, y2))
    return bboxes

def create_mask(img_shape, bboxes):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask

def extract_objects(img, bboxes):
    objects = []
    for (x1, y1, x2, y2) in bboxes:
        obj = img[y1:y2, x1:x2]
        objects.append(((x1, y1), obj))
    return objects

def paste_objects_on_background(bg, objects):
    for (x, y), obj in objects:
        h, w = obj.shape[:2]
        if y + h <= bg.shape[0] and x + w <= bg.shape[1]:  # simple bounds check
            bg[y:y+h, x:x+w] = obj
    return bg

def generate_synthetic_dataset(image_dir, label_dir, bg_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    bg_files = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.endswith(('.jpg', '.png'))]

    for img_file in image_files:
        image_path = os.path.join(image_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            continue

        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]

        bboxes = load_yolo_labels(label_path, img_w, img_h)
        if not bboxes:
            continue

        mask = create_mask(img.shape, bboxes)
        objects = extract_objects(img, bboxes)

        bg_img_path = random.choice(bg_files)
        bg = cv2.imread(bg_img_path)
        bg = cv2.resize(bg, (img_w, img_h))

        new_img = paste_objects_on_background(bg.copy(), objects)

        # Save synthetic image and label in the same directories with prefix
        synthetic_img_name = f"synthetic_{img_file}"
        synthetic_label_name = f"synthetic_{label_file}"

        cv2.imwrite(os.path.join(image_dir, synthetic_img_name), new_img)
        with open(label_path, 'r') as lf, open(os.path.join(label_dir, synthetic_label_name), 'w') as out_lf:
            out_lf.write(lf.read())

        print(f"Generated: {synthetic_img_name} and {synthetic_label_name}")

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset by pasting objects on background images')
    parser.add_argument('--image_dir', type=str, required=True, 
                       help='Directory containing original images')
    parser.add_argument('--label_dir', type=str, required=True,
                       help='Directory containing YOLO label txt files')
    parser.add_argument('--bg_dir', type=str, required=True,
                       help='Directory containing background images')
    
    args = parser.parse_args()
    
    generate_synthetic_dataset(args.image_dir, args.label_dir, args.bg_dir)

if __name__ == "__main__":
    main()
