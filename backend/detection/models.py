import uuid
from django.db import models


class DetectionResult(models.Model):
    """Stores detection results for analytics and history."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to="detections/%Y/%m/%d/")
    results_json = models.JSONField(default=list)
    num_detections = models.IntegerField(default=0)
    max_confidence = models.FloatField(default=0.0)
    processing_time_ms = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "detection_result"
        ordering = ["-created_at"]

    def __str__(self):
        return f"Detection {self.id} - {self.num_detections} found ({self.max_confidence:.2f})"
