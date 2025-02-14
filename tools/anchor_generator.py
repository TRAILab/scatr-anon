import numpy as np
from sklearn.cluster import KMeans
import mmengine

from projects.mmdet3d_plugin.core.box3d import *


def get_kmeans_anchor(
    ann_file,
    num_anchor=900,
    detection_range=55,
    output_file_name="nuscenes_kmeans900.npy",
    verbose=False,
):
    data = mmengine.load(ann_file, file_format="pkl")
    # gt_boxes = np.concatenate([x["gt_boxes"] for x in data["infos"]], axis=0)
    gt_boxes = np.concatenate([
        np.array([instance['bbox_3d'] for instance in data_single["instances"]], dtype=np.float64).reshape(-1,7)
        for data_single in data["data_list"]
    ])

    distance = np.linalg.norm(gt_boxes[:, :3], axis=-1, ord=2)
    mask = distance <= detection_range
    gt_boxes = gt_boxes[mask]
    clf = KMeans(n_clusters=num_anchor, verbose=verbose, random_state=0)
    print("===========Starting kmeans, please wait.===========")
    clf.fit(gt_boxes[:, [X, Y, Z]])
    anchor = np.zeros((num_anchor, 11))
    anchor[:, [X, Y, Z]] = clf.cluster_centers_
    anchor[:, [W, L, H]] = np.log(gt_boxes[:, [W, L, H]].mean(axis=0))
    anchor[:, COS_YAW] = 1
    np.save(output_file_name, anchor)
    print(f"===========Done! Save results to {output_file_name}.===========")


def compute_class_dimensions(ann_file, output_file_name="class_dimensions.npy"):
    data = mmengine.load(ann_file, file_format="pkl")
    print("data loaded")
    gt_boxes = np.concatenate([
        np.array([instance['bbox_3d'] for instance in data_single["instances"]], dtype=np.float64).reshape(-1,7)
        for data_single in data["data_list"]
    ])
    gt_labels = np.concatenate([
        np.array([instance['bbox_label_3d'] for instance in data_single["instances"]], dtype=np.float32).reshape(-1)
        for data_single in data["data_list"]
    ])
    print("gt_boxes and gt_labels concatenated")
    max_cls = gt_labels.max().astype(int)

    class_dimensions = []
    for label in range(max_cls + 1):
        class_boxes = gt_boxes[gt_labels == label]
        if len(class_boxes) == 0:
            class_dimensions.append((0, 0))
        else:
            class_dimensions.append(np.median(class_boxes[:, [W, L]], axis=0))
    class_dimensions = np.array(class_dimensions)
    print("Class dimensions (width, length) for each class:")
    for label, dimensions in enumerate(class_dimensions):
        print(f"Class {label}: Width = {dimensions[0]}, Length = {dimensions[1]}")
    # normalize with log
    class_dimensions = np.log(class_dimensions + np.array([1e-6, 1e-6]))
    np.save(output_file_name, class_dimensions)
    print(
        f"===========Done! Save class dimensions to {output_file_name}.===========")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="anchor kmeans")
    parser.add_argument("--ann_file", type=str, required=True)
    parser.add_argument("--num_anchor", type=int, default=900)
    parser.add_argument("--detection_range", type=float, default=55)
    parser.add_argument(
        "--output_file_name", type=str, default="_nuscenes_kmeans900.npy"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compute_class_dimensions", action="store_true")
    parser.add_argument("--class_output_file_name",
                        type=str, default="avg_class_dimensions.npy")
    args = parser.parse_args()
    if args.compute_class_dimensions:
        compute_class_dimensions(args.ann_file, args.class_output_file_name)
    else:
        get_kmeans_anchor(
            args.ann_file,
            args.num_anchor,
            args.detection_range,
            args.output_file_name,
            args.verbose,
        )
