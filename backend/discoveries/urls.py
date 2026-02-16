from rest_framework.routers import DefaultRouter
from .views import DiscoveryViewSet

router = DefaultRouter()
router.register(r"", DiscoveryViewSet, basename="discovery")

urlpatterns = router.urls
