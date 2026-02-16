from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters, viewsets
from .models import Discovery
from .serializers import DiscoverySerializer


class DiscoveryViewSet(viewsets.ModelViewSet):
    queryset = Discovery.objects.all()
    serializer_class = DiscoverySerializer
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["clover_type", "is_verified"]
    ordering_fields = ["created_at", "confidence"]
    ordering = ["-created_at"]
