from pycocotools.coco import COCO
import json
import os
import random
import shutil


src_img_dir_train = "data/coco/train2017"
src_img_dir_val = "data/coco/val2017"
src_ann_train = "data/coco/annotations/instances_train2017.json"
src_ann_val = "data/coco/annotations/instances_val2017.json"
dst_root = "data/coco_potted"

def create_subset(src_img_dir, src_ann, dst_img_dir, dst_ann, num_images=None):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(dst_ann), exist_ok=True)

    coco = COCO(src_ann)

    # Get category IDs for 'potted plant'
    cat_ids = coco.getCatIds(catNms=['potted plant'])
    if not cat_ids:
        raise ValueError("Category 'potted plant' not found in the dataset.")
    cat_id = cat_ids[0]

    # Get all image IDs containing 'potted plant'
    img_ids = coco.getImgIds(catIds=[cat_id])
    if num_images is not None:
        # Shuffle and select a subset of image IDs if num_images is specified
        random.shuffle(img_ids)
        img_ids = img_ids[:num_images]

    # Build the image list
    imgs = coco.loadImgs(img_ids)

    # Gather annotations for the selected images
    ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=[cat_id], iscrowd=None) # iscrowd determines whether to include crowd annotations
    anns = coco.loadAnns(ann_ids)

    # Categories only include 'potted plant'
    cats = coco.loadCats([cat_id])

    new_ann = {
        "images": imgs,
        "annotations": anns,
        "categories": cats
    }

    # copy images 
    for img in imgs:
        file_name = img["file_name"]
        src_path = os.path.join(src_img_dir, file_name)
        dst_path = os.path.join(dst_img_dir, file_name)
        shutil.copy2(src_path, dst_path)

    # write new annotation file
    with open(dst_ann, 'w') as f:
        json.dump(new_ann, f)


if __name__ == "__main__":
    # Create training subset
    create_subset(
        src_img_dir=src_img_dir_train,
        src_ann=src_ann_train,
        dst_img_dir=os.path.join(dst_root, "train", "images"),
        dst_ann=os.path.join(dst_root, "annotations", "instances_train_potted.json"),
        num_images=1500  # Specify number of images for training subset
    )

    # Create validation subset
    create_subset(
        src_img_dir=src_img_dir_val,
        src_ann=src_ann_val,
        dst_img_dir=os.path.join(dst_root, "val", "images"),
        dst_ann=os.path.join(dst_root, "annotations", "instances_val_potted.json"),
        num_images=300  # Specify number of images for validation subset
    )