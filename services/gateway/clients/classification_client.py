import requests

from config import CLASSIFICATION_SERVICE_URL


def predict_defect(sensor_data):
    """Mengirim data sensor ke service klasifikasi."""
    url = f"{CLASSIFICATION_SERVICE_URL}/predict"

    try:
        response = requests.post(
            url,
            json=sensor_data,
            timeout=10,
        )

        try:
            result = response.json()
        except ValueError:
            result = {
                "error": "Respons dari classification service bukan JSON.",
                "detail": response.text,
            }

        return result, response.status_code

    except requests.RequestException as error:
        return {
            "error": "Service klasifikasi belum dapat dihubungi.",
            "detail": str(error),
        }, 503