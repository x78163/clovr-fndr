from django.contrib import admin
from .models import Discovery


@admin.register(Discovery)
class DiscoveryAdmin(admin.ModelAdmin):
    list_display = ["id", "clover_type", "confidence", "location_name", "is_verified", "created_at"]
    list_filter = ["clover_type", "is_verified", "created_at"]
    search_fields = ["location_name", "notes"]
    ordering = ["-created_at"]
    readonly_fields = ["id", "created_at", "updated_at"]
