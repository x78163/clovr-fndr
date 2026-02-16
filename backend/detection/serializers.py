from rest_framework import serializers


class DetectionRequestSerializer(serializers.Serializer):
    image = serializers.ImageField()
    confidence = serializers.FloatField(required=False, default=0.25, min_value=0.01, max_value=1.0)


class DetectionItemSerializer(serializers.Serializer):
    class_name = serializers.CharField(source="class")
    class_id = serializers.IntegerField()
    confidence = serializers.FloatField()
    bbox = serializers.ListField(child=serializers.FloatField())
    bbox_normalized = serializers.ListField(child=serializers.FloatField())


class DetectionResponseSerializer(serializers.Serializer):
    detections = DetectionItemSerializer(many=True)
    count = serializers.IntegerField()
    processing_time_ms = serializers.FloatField()
    image_size = serializers.ListField(child=serializers.IntegerField())
