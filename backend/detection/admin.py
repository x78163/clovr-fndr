from django.contrib import admin
from .models import DetectionResult


@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = ["id", "num_detections", "max_confidence", "processing_time_ms", "created_at"]
    list_filter = ["created_at"]
    ordering = ["-created_at"]
    readonly_fields = ["id", "created_at"]
