import time

from PIL import Image
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import DetectionRequestSerializer
from .services.yolo_service import get_service


class DetectView(APIView):
    """POST /api/detect/ - Run YOLO detection on an uploaded image."""

    parser_classes = [MultiPartParser]

    def post(self, request):
        serializer = DetectionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        image_file = serializer.validated_data["image"]
        confidence = serializer.validated_data["confidence"]

        # Get image dimensions
        img = Image.open(image_file)
        img_size = [img.width, img.height]
        image_file.seek(0)

        # Run detection
        start = time.perf_counter()
        service = get_service()
        image_bytes = image_file.read()
        detections = service.detect_from_bytes(image_bytes, conf=confidence)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return Response({
            "detections": detections,
            "count": len(detections),
            "processing_time_ms": round(elapsed_ms, 1),
            "image_size": img_size,
        })


class HealthView(APIView):
    """GET /api/detect/health/ - Check YOLO service status."""

    def get(self, request):
        service = get_service()
        return Response(service.health_check())
