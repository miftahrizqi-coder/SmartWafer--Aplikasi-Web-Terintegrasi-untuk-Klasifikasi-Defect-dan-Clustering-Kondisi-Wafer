
from flask import Flask, jsonify, render_template
from flask import Flask, jsonify, render_template, request
from clients.classification_client import predict_defect
from clients.clustering_client import cluster_wafer

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")



@app.route("/klasifikasi", methods=["GET", "POST"])
def klasifikasi():
    if request.method == "POST":
        sensor_data = request.get_json(silent=True)

        if not isinstance(sensor_data, dict):
            return jsonify({
                "error": "Data sensor harus berupa JSON object."
            }), 400

        result, status_code = predict_defect(sensor_data)
        return jsonify(result), status_code

    return render_template("klasifikasi.html")




@app.route("/clustering", methods=["GET", "POST"])
def clustering():
    if request.method == "POST":
        sensor_data = request.get_json(silent=True)

        if not isinstance(sensor_data, dict):
            return jsonify({
                "error": "Data sensor harus berupa JSON object."
            }), 400

        result = cluster_wafer(sensor_data)
        return jsonify(result)

    return render_template("clustering.html")



@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "SmartWafer Gateway"
    })


if __name__ == "__main__":
    from config import HOST, PORT, DEBUG

    app.run(host=HOST, port=PORT, debug=DEBUG)
