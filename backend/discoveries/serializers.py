from rest_framework import serializers
from .models import Discovery


class DiscoverySerializer(serializers.ModelSerializer):
    class Meta:
        model = Discovery
        fields = "__all__"
        read_only_fields = ["id", "created_at", "updated_at"]
