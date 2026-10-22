
import requests

from config import CLUSTERING_SERVICE_URL


def cluster_wafer(sensor_data):
    """Mengirim data sensor ke service clustering."""
    url = f"{CLUSTERING_SERVICE_URL}/predict"

    try:
        response = requests.post(
            url,
            json=sensor_data,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        return {
            "error": "Service clustering belum dapat dihubungi.",
            "detail": str(error),
        }
