from vision.detection_distance import nearest_bbox_median_distance


def test_nearest_valid_bbox_median_distance_is_selected():
    detections = [
        {"class": "red_buoy", "distance": 5.4},
        {"class": "green_buoy", "distance": 2.1},
        {"class": "yellow_buoy", "distance": 3.7},
    ]

    assert nearest_bbox_median_distance(detections) == 2.1


def test_invalid_bbox_median_distances_are_ignored():
    detections = [
        {"distance": None},
        {"distance": float("nan")},
        {"distance": float("inf")},
        {"distance": 0.0},
        {"distance": -1.0},
        "not-a-detection",
    ]

    assert nearest_bbox_median_distance(detections) is None
