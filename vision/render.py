import cv2
import numpy as np


def draw_detections(bgr_image, detections):
    """Draw cached detection dictionaries without importing a detector model."""
    output_frame = bgr_image.copy()
    image_h, image_w = output_frame.shape[:2]

    for detection in detections:
        bbox = detection.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, image_w - 1))
        y1 = max(0, min(y1, image_h - 1))
        x2 = max(0, min(x2, image_w - 1))
        y2 = max(0, min(y2, image_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        class_name = detection.get("class", detection.get("type", "unknown"))
        confidence = detection.get("confidence")
        distance = detection.get("distance")
        track_id = detection.get("track_id")
        label_parts = [str(class_name)]

        if confidence is not None:
            try:
                label_parts.append(f"{float(confidence):.2f}")
            except (TypeError, ValueError):
                pass

        if distance is not None:
            try:
                distance_value = float(distance)
            except (TypeError, ValueError):
                distance_value = float("nan")
            if np.isfinite(distance_value):
                label_parts.append(f"{distance_value:.2f} m")

        angle = None
        side = None
        for key, value in detection.items():
            if key.endswith(" angle: "):
                angle = value
            elif key.endswith(" side: "):
                side = value

        if angle is not None:
            try:
                label_parts.append(f"{float(angle):.1f} deg")
            except (TypeError, ValueError):
                pass
        if side is not None:
            label_parts.append(str(side))
        if track_id is not None:
            label_parts.append(f"ID:{track_id}")

        label = " | ".join(label_parts)
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )
        text_y = max(y1 - 8, text_height + 8)
        cv2.rectangle(
            output_frame,
            (x1, text_y - text_height - 6),
            (x1 + text_width + 6, text_y + baseline),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            output_frame,
            label,
            (x1 + 3, text_y - 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return output_frame
