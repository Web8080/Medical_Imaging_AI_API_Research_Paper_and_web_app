"""
Locust performance testing for Medical Imaging AI API.
"""

from locust import HttpUser, task, between
import random
import io
from PIL import Image


class MedicalImagingAPIUser(HttpUser):
    """Locust user class for testing Medical Imaging AI API."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts."""
        # Test health endpoint first
        self.client.get("/health")
    
    @task(3)
    def test_health_endpoint(self):
        """Test health endpoint (high frequency)."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed with status {response.status_code}")
    
    @task(2)
    def test_models_endpoint(self):
        """Test models endpoint."""
        with self.client.get("/models", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Models endpoint failed with status {response.status_code}")
    
    @task(2)
    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics endpoint failed with status {response.status_code}")
    
    @task(1)
    def test_upload_endpoint(self):
        """Test image upload endpoint (lower frequency due to processing time)."""
        # Create a simple test image
        test_image = self.create_test_image()
        
        files = {
            "file": ("test_image.jpg", test_image, "image/jpeg")
        }
        
        with self.client.post("/upload", files=files, catch_response=True) as response:
            if response.status_code == 200:
                # Check if response contains expected fields
                try:
                    data = response.json()
                    if "prediction" in data or "error" in data:
                        response.success()
                    else:
                        response.failure("Response missing prediction or error field")
                except:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Upload failed with status {response.status_code}")
    
    @task(1)
    def test_root_endpoint(self):
        """Test root endpoint."""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Root endpoint failed with status {response.status_code}")
    
    def create_test_image(self):
        """Create a simple test image for upload testing."""
        # Create a simple 28x28 RGB image
        image = Image.new('RGB', (28, 28), color=(128, 128, 128))
        
        # Add some random noise to make it more realistic
        import random
        pixels = image.load()
        for i in range(28):
            for j in range(28):
                noise = random.randint(-50, 50)
                r = max(0, min(255, 128 + noise))
                g = max(0, min(255, 128 + noise))
                b = max(0, min(255, 128 + noise))
                pixels[i, j] = (r, g, b)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        return img_byte_arr.getvalue()


class HighLoadUser(HttpUser):
    """High load user for stress testing."""
    
    wait_time = between(0.1, 0.5)  # Very short wait time
    
    @task(10)
    def rapid_health_checks(self):
        """Rapid health checks."""
        self.client.get("/health")
    
    @task(5)
    def rapid_metrics_checks(self):
        """Rapid metrics checks."""
        self.client.get("/metrics")


class SlowUser(HttpUser):
    """Slow user for realistic load testing."""
    
    wait_time = between(5, 10)  # Longer wait time
    
    @task(1)
    def slow_upload_test(self):
        """Slow upload test."""
        test_image = self.create_test_image()
        files = {
            "file": ("test_image.jpg", test_image, "image/jpeg")
        }
        self.client.post("/upload", files=files)
    
    def create_test_image(self):
        """Create a test image."""
        image = Image.new('RGB', (28, 28), color=(128, 128, 128))
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue()
