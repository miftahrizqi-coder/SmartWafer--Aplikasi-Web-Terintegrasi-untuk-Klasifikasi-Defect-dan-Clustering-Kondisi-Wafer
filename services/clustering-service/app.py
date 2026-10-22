
import json
import os

import joblib
import numpy as np
from flask import Flask, jsonify, request

from preprocessing import prepare_sensor_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

app = Flask(__name__)

# Muat model dan metadata saat service dijalankan.
bundle = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

pipeline = bundle["pipeline"]
pca = bundle["pca"]
features = bundle["features"]

scaler = pipeline.named_steps["scaler"]
kmeans = pipeline.named_steps["kmeans"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "clustering-service",
        "algorithm": bundle.get("algorithm", "KMeans"),
        "model_version": bundle.get("model_version", "unknown"),
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "error": "Kirim data sensor dalam format JSON."
        }), 400

    try:
        sensor_df = prepare_sensor_data(data)

        # Pastikan urutan fitur sama dengan saat training.
        sensor_df = sensor_df[features]

        # Pipeline melakukan scaling sebelum prediksi.
        scaled_data = scaler.transform(sensor_df)
        cluster_id = int(kmeans.predict(scaled_data)[0])

        centroid = kmeans.cluster_centers_[cluster_id]
        distance = float(
            np.linalg.norm(scaled_data[0] - centroid)
        )

        cluster_profiles = metadata.get("clusters", [])
        cluster_info = next(
            (
                item for item in cluster_profiles
                if int(item.get("cluster_id", -1)) == cluster_id
            ),
            None,
        )

        if cluster_info is None and cluster_id < len(cluster_profiles):
            cluster_info = cluster_profiles[cluster_id]

        if cluster_info is None:
            return jsonify({
                "error": "Profil cluster tidak ditemukan."
            }), 500

        threshold = float(cluster_info["distance_threshold"])
        pca_coordinates = pca.transform(scaled_data)[0]

        return jsonify({
            "cluster_id": cluster_id,
            "cluster_name": cluster_info.get(
                "name", f"Cluster {cluster_id}"
            ),
            "distance_to_centroid": round(distance, 4),
            "distance_threshold": round(threshold, 4),
            "is_anomaly": bool(distance > threshold),
            "pca": {
                "x": round(float(pca_coordinates[0]), 4),
                "y": round(float(pca_coordinates[1]), 4),
            },
        }), 200

    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except (KeyError, TypeError, IndexError) as error:
        app.logger.exception("Kesalahan format model atau metadata")
        return jsonify({
            "error": "Data model atau metadata tidak sesuai.",
            "detail": str(error),
        }), 500
    except Exception:
        app.logger.exception("Prediksi clustering gagal")
        return jsonify({
            "error": "Prediksi clustering gagal."
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
