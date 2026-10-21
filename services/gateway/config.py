
import os

# Alamat service Machine Learning
CLASSIFICATION_SERVICE_URL = os.getenv(
    "CLASSIFICATION_SERVICE_URL",
    "http://127.0.0.1:5001",
)

CLUSTERING_SERVICE_URL = os.getenv(
    "CLUSTERING_SERVICE_URL",
    "http://127.0.0.1:5002",
)

# Konfigurasi server Gateway
HOST = os.getenv("GATEWAY_HOST", "127.0.0.1")
PORT = int(os.getenv("GATEWAY_PORT", "5000"))
DEBUG = os.getenv("GATEWAY_DEBUG", "true").lower() == "true"

# Validasi rentang input sensor
SENSOR_RANGES = {
    "temperature": (0, 150),
    "vibration": (-50, 150),
    "humidity": (0, 100),
    "pressure": (0, 10),
    "energy_consumption": (0, 10),
}
