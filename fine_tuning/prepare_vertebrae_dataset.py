"""
Dataset preparation script for vertebrae pose estimation with RTMPose.

This script helps you prepare a custom COCO-format dataset with vertebrae keypoints
for fine-tuning RTMPose models.

Custom Keypoint Format (24 keypoints total):
- 0-16: Standard COCO keypoints (17 points)
- 17-23: Vertebrae keypoints (7 points):
    17: C7 (base of neck/top of thoracic spine)
    18: T4 (upper thoracic)
    19: T8 (mid thoracic)
    20: T12 (lower thoracic)
    21: L2 (upper lumbar)
    22: L4 (mid lumbar)
    23: S1 (sacrum/top of pelvis)

Usage:
1. Annotate your images using labelme or COCO annotation tools
2. Run this script to validate and convert annotations to MMPose format
3. Use the generated dataset for RTMPose training
"""

import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple


class VertebraeDatasetPreparer:
    """Prepare COCO-format dataset with vertebrae annotations for RTMPose training"""

    # Custom keypoint definition
    KEYPOINT_NAMES = [
        # Standard COCO keypoints (0-16)
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
        # Vertebrae keypoints (17-23)
        'C7_vertebra', 'T4_vertebra', 'T8_vertebra', 'T12_vertebra',
        'L2_vertebra', 'L4_vertebra', 'S1_vertebra'
    ]

    KEYPOINT_SKELETON = [
        # Standard COCO skeleton
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7],
        # Vertebrae skeleton (spine connections)
        [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23],
        # Connect shoulders to C7
        [6, 17], [7, 17],
        # Connect S1 to hips
        [23, 12], [23, 13]
    ]

    NUM_KEYPOINTS = 24

    def __init__(self, output_dir: str = "data/vertebrae_dataset"):
        """
        Initialize dataset preparer

        Args:
            output_dir: Output directory for prepared dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_empty_dataset(self, dataset_name: str = "vertebrae_coco") -> Dict:
        """
        Create an empty COCO-format dataset structure

        Args:
            dataset_name: Name of the dataset

        Returns:
            Empty COCO dataset dictionary
        """
        dataset = {
            "info": {
                "description": f"{dataset_name} - Custom vertebrae pose dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "",
                "date_created": datetime.now().strftime("%Y/%m/%d")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Custom License",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "person",
                    "keypoints": self.KEYPOINT_NAMES,
                    "skeleton": self.KEYPOINT_SKELETON
                }
            ]
        }
        return dataset

    def add_image(self, dataset: Dict, image_path: str, image_id: int) -> Dict:
        """
        Add an image entry to the dataset

        Args:
            dataset: COCO dataset dictionary
            image_path: Path to the image file
            image_id: Unique image ID

        Returns:
            Updated dataset
        """
        from PIL import Image

        img = Image.open(image_path)
        width, height = img.size

        image_entry = {
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height,
            "license": 1
        }

        dataset["images"].append(image_entry)
        return dataset

    def add_annotation(
        self,
        dataset: Dict,
        image_id: int,
        annotation_id: int,
        keypoints: List[float],
        bbox: List[float] = None
    ) -> Dict:
        """
        Add a person annotation with keypoints to the dataset

        Args:
            dataset: COCO dataset dictionary
            image_id: ID of the image this annotation belongs to
            annotation_id: Unique annotation ID
            keypoints: List of keypoint coordinates [x1, y1, v1, x2, y2, v2, ...]
                      where v is visibility (0=not labeled, 1=labeled but occluded, 2=visible)
            bbox: Bounding box [x, y, width, height]. If None, will be computed from keypoints

        Returns:
            Updated dataset
        """
        if len(keypoints) != self.NUM_KEYPOINTS * 3:
            raise ValueError(f"Expected {self.NUM_KEYPOINTS * 3} keypoint values, got {len(keypoints)}")

        # Compute bbox from keypoints if not provided
        if bbox is None:
            bbox = self._compute_bbox_from_keypoints(keypoints)

        # Compute area
        area = bbox[2] * bbox[3]

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "keypoints": keypoints,
            "num_keypoints": sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0),
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        }

        dataset["annotations"].append(annotation)
        return dataset

    def _compute_bbox_from_keypoints(self, keypoints: List[float]) -> List[float]:
        """
        Compute bounding box from keypoints

        Args:
            keypoints: List of keypoint coordinates [x1, y1, v1, x2, y2, v2, ...]

        Returns:
            Bounding box [x, y, width, height]
        """
        visible_keypoints = []
        for i in range(0, len(keypoints), 3):
            if keypoints[i + 2] > 0:  # visibility > 0
                visible_keypoints.append([keypoints[i], keypoints[i + 1]])

        if not visible_keypoints:
            return [0, 0, 0, 0]

        visible_keypoints = np.array(visible_keypoints)
        x_min, y_min = visible_keypoints.min(axis=0)
        x_max, y_max = visible_keypoints.max(axis=0)

        # Add padding
        padding = 10
        width = x_max - x_min + 2 * padding
        height = y_max - y_min + 2 * padding

        return [float(x_min - padding), float(y_min - padding), float(width), float(height)]

    def save_dataset(self, dataset: Dict, split: str = "train"):
        """
        Save dataset to JSON file

        Args:
            dataset: COCO dataset dictionary
            split: Dataset split name (train/val/test)
        """
        output_path = self.output_dir / f"{split}.json"
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"Saved {split} dataset to {output_path}")
        print(f"  Images: {len(dataset['images'])}")
        print(f"  Annotations: {len(dataset['annotations'])}")

    def validate_annotation(self, keypoints: List[float]) -> Tuple[bool, str]:
        """
        Validate keypoint annotation

        Args:
            keypoints: List of keypoint coordinates [x1, y1, v1, x2, y2, v2, ...]

        Returns:
            (is_valid, error_message)
        """
        if len(keypoints) != self.NUM_KEYPOINTS * 3:
            return False, f"Expected {self.NUM_KEYPOINTS * 3} values, got {len(keypoints)}"

        # Check that at least some keypoints are visible
        num_visible = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)
        if num_visible == 0:
            return False, "No visible keypoints found"

        # Check that vertebrae keypoints are in order (when visible)
        vertebrae_indices = [17, 18, 19, 20, 21, 22, 23]  # C7 to S1
        vertebrae_y = []
        for idx in vertebrae_indices:
            vis = keypoints[idx * 3 + 2]
            if vis > 0:
                y = keypoints[idx * 3 + 1]
                vertebrae_y.append(y)

        # Vertebrae should be ordered from top (low y) to bottom (high y)
        if len(vertebrae_y) > 1:
            if not all(vertebrae_y[i] <= vertebrae_y[i+1] for i in range(len(vertebrae_y)-1)):
                return False, "Vertebrae keypoints are not in anatomical order (top to bottom)"

        return True, ""

    def create_annotation_template(self) -> Dict:
        """
        Create a template annotation with all keypoints set to not labeled

        Returns:
            Template keypoints array with all visibility set to 0
        """
        keypoints = [0.0, 0.0, 0.0] * self.NUM_KEYPOINTS
        return keypoints


def create_example_dataset():
    """
    Create an example dataset with dummy data to demonstrate the format
    """
    preparer = VertebraeDatasetPreparer(output_dir="data/vertebrae_example")

    # Create empty datasets
    train_dataset = preparer.create_empty_dataset("vertebrae_train")
    val_dataset = preparer.create_empty_dataset("vertebrae_val")

    print("\nExample dataset structure created!")
    print("\nKeypoint indices:")
    for idx, name in enumerate(preparer.KEYPOINT_NAMES):
        print(f"  {idx}: {name}")

    print("\nSkeleton connections:")
    for connection in preparer.KEYPOINT_SKELETON:
        kp1, kp2 = connection
        print(f"  {preparer.KEYPOINT_NAMES[kp1]} <-> {preparer.KEYPOINT_NAMES[kp2]}")

    # Save empty datasets
    preparer.save_dataset(train_dataset, "train")
    preparer.save_dataset(val_dataset, "val")

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("\n1. Annotate your cat-cow exercise images with 24 keypoints")
    print("   - Use labelme, CVAT, or any COCO annotation tool")
    print("   - Mark all 17 COCO keypoints + 7 vertebrae points")
    print("\n2. Convert your annotations to COCO format:")
    print("   - Use this script's add_image() and add_annotation() methods")
    print("\n3. Place annotated images in: data/vertebrae_dataset/images/")
    print("\n4. Run the training script (train_rtmpose_vertebrae.py)")
    print("\n" + "="*60)


if __name__ == "__main__":
    create_example_dataset()