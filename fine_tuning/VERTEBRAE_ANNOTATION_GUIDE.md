# Vertebrae Annotation Guide for Cat-Cow Exercise Analysis

This guide explains how to annotate vertebrae keypoints for training a custom RTMPose model.

## Overview

The custom pose model detects **24 keypoints**:
- **17 standard COCO keypoints** (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
- **7 vertebrae keypoints** (C7, T4, T8, T12, L2, L4, S1)

## Keypoint Definitions

### Vertebrae Keypoints (17-23)

| Index | Name | Description | Anatomical Location |
|-------|------|-------------|---------------------|
| 17 | C7_vertebra | Cervical vertebra 7 | Base of neck, top of thoracic spine (most prominent bump at neck base) |
| 18 | T4_vertebra | Thoracic vertebra 4 | Upper thoracic, approximately at armpit level |
| 19 | T8_vertebra | Thoracic vertebra 8 | Mid thoracic, approximately at lower scapula |
| 20 | T12_vertebra | Thoracic vertebra 12 | Lower thoracic, at bottom of rib cage |
| 21 | L2_vertebra | Lumbar vertebra 2 | Upper lumbar, mid-lower back |
| 22 | L4_vertebra | Lumbar vertebra 4 | Mid lumbar, lower back |
| 23 | S1_vertebra | Sacral vertebra 1 | Top of sacrum/pelvis, at hip level |

### Visual Reference for Side View (Profile)

```
        Head (nose, eyes, ears)
           |
    C7 ----+---- (neck base)
           |
    T4 ----+---- (upper back)
           |
    T8 ----+---- (mid back)
           |
   T12 ----+---- (lower back)
           |
    L2 ----+---- (lumbar upper)
           |
    L4 ----+---- (lumbar mid)
           |
    S1 ----+---- (sacrum/pelvis)
           |
         Hips
```

## Annotation Requirements

### Camera Position
- **Side view (profile)** is ESSENTIAL for vertebrae visibility
- Camera perpendicular to subject's body
- Full torso visible from head to hips
- Subject should be on all fours (cat-cow position)

### Minimum Annotated Keypoints
For each image, you MUST annotate at minimum:
- **Left shoulder** (5)
- **Right shoulder** (6)
- **Left hip** (11)
- **Right hip** (12)
- **All 7 vertebrae points** (17-23)

Additional COCO keypoints (nose, elbows, knees, etc.) can improve model performance but are optional.

### Keypoint Visibility Codes
- **0**: Not labeled (point not visible or unknown)
- **1**: Labeled but occluded (point exists but hidden)
- **2**: Labeled and visible (point clearly visible)

## Annotation Tools

### Option 1: Labelme (Recommended for beginners)

```bash
pip install labelme
labelme images/ --labels vertebrae_labels.txt
```

Create `vertebrae_labels.txt`:
```
nose
left_eye
right_eye
left_ear
right_ear
left_shoulder
right_shoulder
left_elbow
right_elbow
left_wrist
right_wrist
left_hip
right_hip
left_knee
right_knee
left_ankle
right_ankle
C7_vertebra
T4_vertebra
T8_vertebra
T12_vertebra
L2_vertebra
L4_vertebra
S1_vertebra
```

### Option 2: CVAT (Recommended for batch annotation)

1. Create CVAT project: https://app.cvat.ai
2. Import custom skeleton with 24 keypoints
3. Annotate frames with keypoint tool
4. Export in COCO Keypoints format

### Option 3: Custom Python Script

Use the provided `prepare_vertebrae_dataset.py` script to programmatically add annotations.

## Annotation Best Practices

### 1. Anatomical Accuracy
- Vertebrae should form a smooth curve from C7 to S1
- Maintain proper anatomical ordering (top to bottom)
- In cat pose: spine curves UP (convex)
- In cow pose: spine curves DOWN (concave)

### 2. Consistency
- Use the same anatomical landmarks across all images
- If unsure about exact vertebra location, estimate based on proportions:
  - C7: at neck base (shoulder line)
  - T4: ~15% down from C7 to hips
  - T8: ~40% down from C7 to hips
  - T12: ~65% down from C7 to hips
  - L2: ~80% down from C7 to hips
  - L4: ~90% down from C7 to hips
  - S1: at hip line

### 3. Quality Control
- Validate annotations with the script:
```python
from prepare_vertebrae_dataset import VertebraeDatasetPreparer

preparer = VertebraeDatasetPreparer()
is_valid, error = preparer.validate_annotation(keypoints)
if not is_valid:
    print(f"Invalid annotation: {error}")
```

## Dataset Organization

Structure your dataset as follows:

```
data/vertebrae_dataset/
├── images/
│   ├── img_0001.jpg
│   ├── img_0002.jpg
│   └── ...
├── train.json          # COCO format annotations
├── val.json            # COCO format annotations
└── test.json           # COCO format annotations (optional)
```

## Annotation Targets

### Minimum Dataset Size
- **Training set**: 500-1000 annotated images
- **Validation set**: 100-200 annotated images
- **Test set**: 50-100 annotated images (optional)

### Diversity Requirements
Annotate images with variation in:
- **Subjects**: Multiple people with different body types
- **Poses**: Cat, cow, neutral, and transition poses
- **Angles**: Primarily side view, but include slight variations
- **Lighting**: Various lighting conditions
- **Backgrounds**: Different environments

## Converting Annotations to COCO Format

After annotating with labelme or CVAT, convert to COCO format:

```python
from prepare_vertebrae_dataset import VertebraeDatasetPreparer

preparer = VertebraeDatasetPreparer(output_dir="data/vertebrae_dataset")

# Create dataset
dataset = preparer.create_empty_dataset("vertebrae_train")

# Add your annotated images
# (This depends on your annotation tool's output format)

# Save
preparer.save_dataset(dataset, "train")
```

## Training the Model

Once you have prepared your dataset:

```bash
# Single GPU
python train_rtmpose_vertebrae.py

# Multi-GPU (4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 train_rtmpose_vertebrae.py --launcher pytorch

# Resume from checkpoint
python train_rtmpose_vertebrae.py --resume work_dirs/rtmpose_vertebrae/latest.pth
```

## Validation and Testing

Monitor training metrics:
- **AP (Average Precision)**: Overall keypoint detection accuracy
- **AP50**: Precision at 50% IoU threshold
- **AR (Average Recall)**: Detection recall rate

Target metrics for good performance:
- AP > 0.80 for COCO keypoints
- AP > 0.70 for vertebrae keypoints (harder to detect)

## Troubleshooting

### Low Vertebrae Detection Accuracy
- Ensure annotations are anatomically consistent
- Check that vertebrae are visible in profile view
- Increase dataset size (especially vertebrae examples)
- Verify vertebrae ordering (C7 at top, S1 at bottom)

### Model Not Learning
- Check data augmentation (may be too aggressive)
- Verify dataset paths in config file
- Reduce learning rate
- Increase training epochs

### Overfitting
- Add more training data
- Increase data augmentation
- Add dropout or regularization
- Use early stopping based on validation loss

## Additional Resources

- **MMPose Documentation**: https://mmpose.readthedocs.io
- **RTMPose Paper**: https://arxiv.org/abs/2303.07399
- **COCO Keypoint Format**: https://cocodataset.org/#format-data
- **Anatomy Reference**: Use medical anatomy charts for accurate vertebrae placement

## Citation

If you use this annotation guide or training pipeline, please cite:

```bibtex
@article{rtmpose2023,
  title={RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
  author={Jiang, Tao and Lu, Peng and Zhang, Li and Ma, Ningsheng and Han, Rui and Lyu, Chengqi and Li, Yining and Chen, Kai},
  journal={arXiv preprint arXiv:2303.07399},
  year={2023}
}
```