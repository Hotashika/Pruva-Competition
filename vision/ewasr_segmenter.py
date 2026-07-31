"""Runtime adapter for an exported eWaSR TorchScript segmentation model.

The runtime intentionally does not vendor the eWaSR training repository.
Exported TorchScript keeps the live Njord process on its existing PyTorch
dependency while the original model source is needed only during export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


OBSTACLE_CLASS_ID = 0
WATER_CLASS_ID = 1
SKY_CLASS_ID = 2
SEMANTIC_CLASS_NAMES = ("obstacle", "water", "sky")


@dataclass(frozen=True)
class EWaSRResult:
    """One full-resolution three-class eWaSR prediction."""

    label_map: np.ndarray
    confidence_map: np.ndarray

    def __post_init__(self):
        if self.label_map.ndim != 2:
            raise ValueError("eWaSR label_map must be two-dimensional")
        if self.confidence_map.shape != self.label_map.shape:
            raise ValueError(
                "eWaSR confidence_map dimensions must match label_map"
            )

    @property
    def obstacle_mask(self) -> np.ndarray:
        return self.label_map == OBSTACLE_CLASS_ID

    @property
    def water_mask(self) -> np.ndarray:
        return self.label_map == WATER_CLASS_ID

    @property
    def sky_mask(self) -> np.ndarray:
        return self.label_map == SKY_CLASS_ID


class TorchScriptEWaSRBackend:
    """Run the stable two-input TorchScript contract produced by our exporter."""

    def __init__(
        self,
        model_path,
        *,
        device=None,
        input_height=384,
        input_width=512,
        half_precision=True,
    ):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required to load the eWaSR TorchScript model"
            ) from exc

        self.torch = torch
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("eWaSR input dimensions must be positive")

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.half_precision = bool(
            half_precision and self.device.type == "cuda"
        )
        self.model = torch.jit.load(
            str(Path(model_path).expanduser().resolve()),
            map_location=self.device,
        )
        self.model = self.model.eval().to(self.device)
        if self.half_precision:
            self.model = self.model.half()

    def predict(
        self,
        bgr_image: np.ndarray,
        imu_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        functional = torch.nn.functional

        if (
            not isinstance(bgr_image, np.ndarray)
            or bgr_image.ndim != 3
            or bgr_image.shape[2] < 3
        ):
            raise ValueError("eWaSR input must be an HxWx3 BGR image")

        height, width = bgr_image.shape[:2]
        imu_mask = np.asarray(imu_mask)
        if imu_mask.shape != (height, width):
            raise ValueError("eWaSR IMU mask must match the input image")

        rgb = np.ascontiguousarray(bgr_image[:, :, :3][:, :, ::-1])
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0).to(
            self.device,
            non_blocking=True,
        )
        image_tensor = image_tensor.float().div_(255.0)
        image_tensor = functional.interpolate(
            image_tensor,
            size=(self.input_height, self.input_width),
            mode="bilinear",
            align_corners=False,
        )

        mean = image_tensor.new_tensor(
            (0.485, 0.456, 0.406)
        ).view(1, 3, 1, 1)
        std = image_tensor.new_tensor(
            (0.229, 0.224, 0.225)
        ).view(1, 3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        imu_tensor = torch.from_numpy(
            np.ascontiguousarray(imu_mask.astype(np.float32, copy=False))
        )
        imu_tensor = imu_tensor.unsqueeze(0).unsqueeze(0).to(
            self.device,
            non_blocking=True,
        )
        imu_tensor = functional.interpolate(
            imu_tensor,
            size=(self.input_height, self.input_width),
            mode="nearest",
        ).squeeze(1)

        if self.half_precision:
            image_tensor = image_tensor.half()
            imu_tensor = imu_tensor.half()

        with torch.no_grad():
            logits = self.model(image_tensor, imu_tensor)
            if isinstance(logits, dict):
                logits = logits["out"]
            elif isinstance(logits, (tuple, list)):
                logits = logits[0]

            if getattr(logits, "ndim", None) != 4 or logits.shape[1] != 3:
                raise RuntimeError(
                    "eWaSR TorchScript output must have shape [N, 3, H, W]"
                )

            logits = functional.interpolate(
                logits.float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            probabilities = torch.softmax(logits, dim=1)
            confidence, labels = probabilities.max(dim=1)

        return (
            labels[0].byte().cpu().numpy(),
            confidence[0].float().cpu().numpy(),
        )


class OnnxEWaSRBackend:
    """Run official IMU or non-IMU eWaSR ONNX releases."""

    def __init__(
        self,
        model_path,
        *,
        device=None,
        input_height=384,
        input_width=512,
        preserve_aspect_ratio=False,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required to load an eWaSR ONNX model"
            ) from exc

        available = set(ort.get_available_providers())
        wants_cuda = device is not None and "cuda" in str(device).lower()
        providers = []
        if wants_cuda and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(
            str(Path(model_path).expanduser().resolve()),
            providers=providers,
        )

        inputs = self.session.get_inputs()
        if len(inputs) not in (1, 2):
            raise RuntimeError(
                "eWaSR ONNX model must have one image input and an optional "
                f"IMU input; found {len(inputs)}"
            )
        self.image_input = next(
            (
                item
                for item in inputs
                if "image" in item.name.lower()
                or (
                    len(item.shape) == 4
                    and item.shape[1] == 3
                )
            ),
            inputs[0],
        )
        self.imu_input = next(
            (
                item
                for item in inputs
                if item.name != self.image_input.name
            ),
            None,
        )
        self.preserve_aspect_ratio = bool(preserve_aspect_ratio)
        image_shape = self.image_input.shape
        self.input_height = (
            int(image_shape[2])
            if len(image_shape) == 4
            and isinstance(image_shape[2], int)
            and image_shape[2] > 0
            else int(input_height)
        )
        self.input_width = (
            int(image_shape[3])
            if len(image_shape) == 4
            and isinstance(image_shape[3], int)
            and image_shape[3] > 0
            else int(input_width)
        )
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("eWaSR input dimensions must be positive")

    @staticmethod
    def _resize_plane(array, shape, *, nearest=False):
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required for eWaSR ONNX preprocessing"
            ) from exc

        height, width = shape
        image = Image.fromarray(
            np.asarray(array, dtype=np.float32),
            mode="F",
        )
        resampling = (
            Image.Resampling.NEAREST
            if nearest
            else Image.Resampling.BILINEAR
        )
        return np.asarray(
            image.resize((width, height), resampling),
            dtype=np.float32,
        )

    def predict(
        self,
        bgr_image: np.ndarray,
        imu_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required for eWaSR ONNX preprocessing"
            ) from exc

        if (
            not isinstance(bgr_image, np.ndarray)
            or bgr_image.ndim != 3
            or bgr_image.shape[2] < 3
        ):
            raise ValueError("eWaSR input must be an HxWx3 BGR image")

        height, width = bgr_image.shape[:2]
        imu_mask = np.asarray(imu_mask)
        if imu_mask.shape != (height, width):
            raise ValueError("eWaSR IMU mask must match the input image")

        rgb = np.ascontiguousarray(bgr_image[:, :, :3][:, :, ::-1])
        content_box = (0, 0, self.input_width, self.input_height)
        if self.preserve_aspect_ratio:
            scale = min(
                self.input_width / width,
                self.input_height / height,
            )
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            left = (self.input_width - resized_width) // 2
            top = (self.input_height - resized_height) // 2
            resized = Image.fromarray(rgb, mode="RGB").resize(
                (resized_width, resized_height),
                Image.Resampling.BILINEAR,
            )
            canvas = Image.new(
                "RGB",
                (self.input_width, self.input_height),
                color=(0, 0, 0),
            )
            canvas.paste(resized, (left, top))
            resized_rgb = np.asarray(canvas, dtype=np.float32)
            content_box = (
                left,
                top,
                left + resized_width,
                top + resized_height,
            )
        else:
            resized_rgb = np.asarray(
                Image.fromarray(rgb, mode="RGB").resize(
                    (self.input_width, self.input_height),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            )
        image_tensor = resized_rgb.transpose(2, 0, 1)[None] / 255.0
        mean = np.asarray(
            (0.485, 0.456, 0.406),
            dtype=np.float32,
        )[None, :, None, None]
        std = np.asarray(
            (0.229, 0.224, 0.225),
            dtype=np.float32,
        )[None, :, None, None]
        image_tensor = np.ascontiguousarray(
            (image_tensor - mean) / std,
            dtype=np.float32,
        )
        inputs = {self.image_input.name: image_tensor}
        if self.imu_input is not None:
            if self.preserve_aspect_ratio:
                left, top, right, bottom = content_box
                resized_imu = self._resize_plane(
                    imu_mask,
                    (bottom - top, right - left),
                    nearest=True,
                )
                imu_canvas = np.zeros(
                    (self.input_height, self.input_width),
                    dtype=np.float32,
                )
                imu_canvas[top:bottom, left:right] = resized_imu
                imu_tensor = imu_canvas[None]
            else:
                imu_tensor = self._resize_plane(
                    imu_mask,
                    (self.input_height, self.input_width),
                    nearest=True,
                )[None]
            inputs[self.imu_input.name] = np.ascontiguousarray(
                imu_tensor,
                dtype=np.float32,
            )

        outputs = self.session.run(None, inputs)
        logits = next(
            (
                np.asarray(output)
                for output in outputs
                if np.asarray(output).ndim == 4
                and np.asarray(output).shape[1] == 3
            ),
            None,
        )
        if logits is None:
            shapes = [tuple(np.asarray(output).shape) for output in outputs]
            raise RuntimeError(
                "eWaSR ONNX output does not contain [N,3,H,W] logits; "
                f"found {shapes}"
            )

        logits = logits[0].astype(np.float32, copy=False)
        if self.preserve_aspect_ratio:
            output_height, output_width = logits.shape[1:]
            left, top, right, bottom = content_box
            output_left = int(round(left * output_width / self.input_width))
            output_top = int(round(top * output_height / self.input_height))
            output_right = int(
                round(right * output_width / self.input_width)
            )
            output_bottom = int(
                round(bottom * output_height / self.input_height)
            )
            output_left = min(max(output_left, 0), output_width - 1)
            output_top = min(max(output_top, 0), output_height - 1)
            output_right = min(
                max(output_right, output_left + 1),
                output_width,
            )
            output_bottom = min(
                max(output_bottom, output_top + 1),
                output_height,
            )
            logits = logits[
                :,
                output_top:output_bottom,
                output_left:output_right,
            ]
        if logits.shape[1:] != (height, width):
            logits = np.stack(
                [
                    self._resize_plane(channel, (height, width))
                    for channel in logits
                ],
                axis=0,
            )
        logits -= np.max(logits, axis=0, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=0, keepdims=True)
        labels = np.argmax(probabilities, axis=0).astype(np.uint8)
        confidence = np.max(probabilities, axis=0).astype(np.float32)
        return labels, confidence


class EWaSRSegmenter:
    """Fail-open eWaSR facade used by the Task 2 fusion detector."""

    def __init__(
        self,
        model_path=None,
        *,
        enabled=True,
        device=None,
        input_height=384,
        input_width=512,
        half_precision=True,
        preserve_aspect_ratio=False,
        backend: Any = None,
    ):
        self.model_path = None if model_path is None else Path(model_path)
        self.enabled = bool(enabled)
        self.last_error: str | None = None
        self.backend = backend

        if not self.enabled:
            self.last_error = "eWaSR is disabled by configuration"
            return
        if self.backend is not None:
            return
        if self.model_path is None or not self.model_path.is_file():
            self.last_error = (
                "eWaSR model is missing: "
                f"{self.model_path}"
            )
            return

        try:
            if self.model_path.suffix.lower() == ".onnx":
                self.backend = OnnxEWaSRBackend(
                    self.model_path,
                    device=device,
                    input_height=input_height,
                    input_width=input_width,
                    preserve_aspect_ratio=preserve_aspect_ratio,
                )
            else:
                self.backend = TorchScriptEWaSRBackend(
                    self.model_path,
                    device=device,
                    input_height=input_height,
                    input_width=input_width,
                    half_precision=half_precision,
                )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.last_error = str(exc)
            self.backend = None

    @property
    def ready(self) -> bool:
        return self.enabled and self.backend is not None

    def detect(
        self,
        bgr_image: np.ndarray,
        imu_mask: np.ndarray,
    ) -> EWaSRResult | None:
        if not self.ready:
            return None

        try:
            labels, confidence = self.backend.predict(
                bgr_image,
                imu_mask,
            )
            result = EWaSRResult(
                label_map=np.asarray(labels, dtype=np.uint8),
                confidence_map=np.asarray(confidence, dtype=np.float32),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.last_error = str(exc)
            return None

        self.last_error = None
        return result
