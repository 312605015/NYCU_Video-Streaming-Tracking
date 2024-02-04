import os
import json

def convert_labels_to_coco_format(input_labels_dir, output_coco_dir, mode):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "car"}],  
    }

    image_folder = os.path.join(output_coco_dir, mode)
    os.makedirs(image_folder, exist_ok=True)

    label_files = os.listdir(input_labels_dir)
    for idx, label_file in enumerate(label_files):
        with open(os.path.join(input_labels_dir, label_file), 'r') as f:
            lines = f.readlines()

        image_id = idx
        image_info = {
            "id": image_id,
            "file_name": label_file.replace(".txt", ".jpg"),
            "width": 1920,  
            "height": 1080,  
        }
        coco_data["images"].append(image_info)

        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x_min = (x_center - width/2) * 1920  
            y_min = (y_center - height/2) * 1080
            x_max = (x_center + width/2) * 1920
            y_max = (y_center + height/2) * 1080
            annotation_info = {
                "id": len(coco_data["annotations"]),
                "image_id": image_id,
                "category_id": int(class_id),
                "bbox": [x_min, y_min, x_max-x_min, y_max-y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "iscrowd": 0,
                "segmentation": [],
            }
            coco_data["annotations"].append(annotation_info)

    with open(os.path.join(output_coco_dir, f"{mode}_coco_format.json"), "w") as json_file:
        json.dump(coco_data, json_file)


convert_labels_to_coco_format("C:\\software\\python\\HW2_ObjectDetection_2023\\train_labels", "C:\\software\\python\\HW2_ObjectDetection_2023", "train_coco_format")
convert_labels_to_coco_format("C:\\software\\python\\HW2_ObjectDetection_2023\\val_labels", "C:\\software\\python\\HW2_ObjectDetection_2023", "validation_coco_format")
