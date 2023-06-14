import os
import traceback
import numpy as np
import tensorflow_text
import tensorflow as tf

from pydantic import BaseModel
from flask import Flask, Response, request

app = Flask(__name__)

model = tf.keras.models.load_model("model.h5")


@app.route("/")
def index():
    return "Hello world from ML endpoint!"


class RequestText(BaseModel):
    text: str


@app.route("/predict_text", methods=["POST"])
def predict_text():
    try:
        req = request.get_json(force=True)
        text = req["text"]
        print("Uploaded text:", text)

        def preprocess_text(text):
            processed_text = text.lower()
            return processed_text

        def prepare_data(input_data):
            prepared_data = preprocess_text(input_data)
            return [prepared_data]

        def predict_data(data):
            result = model(tf.constant(data))
            return result

        def format_output(result):
            labels = [
                'Teknik Informatika, Sistem Informasi, Ilmu Komputer',
                'Ekonomi, Akuntansi, Manajemen',
                'Seni, Desain Komunikasi Visual, Desain Produk',
                'Kedokteran, Kesehatan Masyarakat, Keperawatan'
            ]
            predicted_index = np.argmax(result)
            output = {"predicted_jurusan": labels[predicted_index]}
            return output

        preprocessed_text = preprocess_text(text)
        data = prepare_data(preprocessed_text)
        prediction = predict_data(data)
        output = format_output(prediction)

        return {"prediction": output}

    except Exception as e:
        traceback.print_exc()
        return "Internal Server Error", 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 4000)))
