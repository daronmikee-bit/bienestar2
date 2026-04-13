from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Función para crear features adicionales
def crear_features(X):
    X = X.copy()
    X['actividad2'] = X['actividad'] ** 2
    X['interaccion'] = X['actividad'] * X['sueno'] * X['estres']
    return X

import sys
sys.modules['__main__'].crear_features = crear_features

# Cargar modelo
modelo = joblib.load("modelo_bienestar.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No se enviaron datos"}), 400

        # Validar todas las variables necesarias
        for var in ["sueno", "actividad", "estres"]:
            if var not in data:
                return jsonify({"error": f"Falta la variable: '{var}'"}), 400
            if not isinstance(data[var], (int, float)):
                return jsonify({"error": f"'{var}' debe ser numérico"}), 400

        df = pd.DataFrame([{
            "actividad": data["actividad"],
            "sueno": data["sueno"],
            "estres": data["estres"]
        }])

        # Crear features adicionales si el modelo los requiere
        df = crear_features(df)

        # Hacer predicción
        pred = modelo.predict(df)[0]

        return jsonify({
            "entrada": data,
            "prediccion": round(pred, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
