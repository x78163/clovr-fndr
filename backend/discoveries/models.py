import uuid
from django.db import models


class Discovery(models.Model):
    """A saved clover discovery with detection metadata."""

    CLOVER_TYPES = [
        ("three-leaf", "3-Leaf Clover"),
        ("four-leaf", "4-Leaf Clover"),
        ("five-leaf", "5-Leaf Clover"),
        ("six-plus-leaf", "6+ Leaf Clover"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to="discoveries/%Y/%m/%d/")
    annotated_image = models.ImageField(
        upload_to="discoveries/annotated/%Y/%m/%d/", null=True, blank=True
    )
    clover_type = models.CharField(max_length=20, choices=CLOVER_TYPES)
    confidence = models.FloatField()
    bbox_json = models.JSONField(default=dict)

    # Location (optional)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    location_name = models.CharField(max_length=200, blank=True)

    # Metadata
    notes = models.TextField(blank=True)
    is_verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "discovery"
        ordering = ["-created_at"]
        verbose_name_plural = "discoveries"

    def __str__(self):
        return f"{self.get_clover_type_display()} ({self.confidence:.2f}) - {self.created_at:%Y-%m-%d}"
