import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import warnings
from main import app

client = TestClient(app)

# Suppress all warnings
warnings.filterwarnings("ignore")


@pytest.fixture
def sample_image():
    # Create a sample image for testing
    image = Image.new("RGB", (224, 224), color="white")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)
    return image_bytes

def test_predict_image(sample_image):
    # Test the predict_image endpoint with a sample image
    files = {"file": ("image.jpg", sample_image, "image/jpeg")}
    response = client.post("/predict/", files=files)
    data = response.json()
    assert "predictions" in data
    predictions = data["predictions"]
    assert isinstance(predictions, list)
    assert len(predictions) == 5  # Assuming topk=5 in the API endpoint
    assert response.status_code == 200

# Additional tests can be added for error cases, edge cases, etc.

