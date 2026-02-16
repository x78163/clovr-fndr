"""YOLO + SAM Inference Service.

YOLO detects bounding boxes, SAM refines them into pixel-level masks.
Thread-safe singleton for Django's multi-threaded request handling.
"""

import logging
import threading
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "three-leaf", 1: "four-leaf", 2: "five-leaf", 3: "six-plus-leaf"}

SAM_MODEL = getattr(settings, "SAM_MODEL", "sam2_b.pt")


class YOLOService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._model = None
                    instance._model_path = None
                    instance._sam = None
                    cls._instance = instance
        return cls._instance

    def _load_model(self):
        """Load the YOLO model from configured path."""
        from ultralytics import YOLO

        model_path = settings.YOLO_MODEL_PATH

        if not Path(model_path).exists():
            logger.warning(f"Model not found at {model_path}. Detection will not work.")
            return

        logger.info(f"Loading YOLO model from {model_path}")
        self._model = YOLO(model_path)
        self._model_path = model_path
        logger.info("YOLO model loaded successfully")

    def _load_sam(self):
        """Load SAM model for segmentation."""
        from ultralytics import SAM

        logger.info(f"Loading SAM model: {SAM_MODEL}")
        self._sam = SAM(SAM_MODEL)
        logger.info("SAM model loaded successfully")

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load_model()
        return self._model

    @property
    def sam(self):
        if self._sam is None:
            with self._lock:
                if self._sam is None:
                    self._load_sam()
        return self._sam

    def _mask_to_polygon(self, mask: np.ndarray, simplify: float = 2.0) -> list[list[float]]:
        """Convert a binary mask to a simplified polygon (list of [x, y] points)."""
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)
        # Simplify to reduce point count for the frontend
        epsilon = simplify * cv2.arcLength(contour, True) / 100
        approx = cv2.approxPolyDP(contour, epsilon, True)

        return [[round(float(pt[0][0]), 1), round(float(pt[0][1]), 1)] for pt in approx]

    def detect(
        self,
        image_source,
        conf: float | None = None,
        iou: float = 0.45,
        imgsz: int = 1280,
        segment: bool = True,
    ) -> list[dict]:
        """Run YOLO detection, optionally refined with SAM segmentation.

        Args:
            image_source: numpy array or PIL Image
            conf: confidence threshold (uses settings default if None)
            iou: IoU threshold for NMS
            imgsz: inference image size
            segment: if True, run SAM on detected bboxes to get masks
        """
        if self.model is None:
            return []

        if conf is None:
            conf = settings.YOLO_CONFIDENCE_THRESHOLD

        results = self.model.predict(
            source=image_source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )

        detections = []
        all_bboxes = []

        for result in results:
            img_h, img_w = result.orig_shape

            for box in result.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                all_bboxes.append([x1, y1, x2, y2])

                detections.append({
                    "class": CLASS_NAMES.get(cls_id, f"class-{cls_id}"),
                    "class_id": cls_id,
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "bbox_normalized": [
                        round(((x1 + x2) / 2) / img_w, 6),
                        round(((y1 + y2) / 2) / img_h, 6),
                        round((x2 - x1) / img_w, 6),
                        round((y2 - y1) / img_h, 6),
                    ],
                    "mask": None,
                })

        # Run SAM segmentation on detected bboxes
        if segment and detections and all_bboxes:
            try:
                sam_results = self.sam(
                    image_source,
                    bboxes=all_bboxes,
                    verbose=False,
                )
                for sam_result in sam_results:
                    if sam_result.masks is not None:
                        for i, mask_tensor in enumerate(sam_result.masks.data):
                            if i < len(detections):
                                mask_np = mask_tensor.cpu().numpy()
                                polygon = self._mask_to_polygon(mask_np)
                                if polygon:
                                    detections[i]["mask"] = polygon
            except Exception as e:
                logger.warning(f"SAM segmentation failed, falling back to bboxes: {e}")

        return detections

    def detect_from_bytes(self, image_bytes: bytes, **kwargs) -> list[dict]:
        """Run inference on image bytes."""
        from PIL import Image
        import io

        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image)
        return self.detect(img_array, **kwargs)

    def health_check(self) -> dict:
        """Return model status info."""
        return {
            "model_loaded": self._model is not None,
            "model_path": str(self._model_path) if self._model_path else None,
            "sam_loaded": self._sam is not None,
            "classes": CLASS_NAMES,
            "default_confidence": settings.YOLO_CONFIDENCE_THRESHOLD,
        }


def get_service() -> YOLOService:
    """Get the singleton YOLOService instance."""
    return YOLOService()
