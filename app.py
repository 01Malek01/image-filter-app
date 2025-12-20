import base64
import os
import secrets

import cv2
import numpy as np
from flask import Flask, flash, redirect, render_template, request, session, url_for

from filters import (
    apply_mean,
    apply_median,
    apply_min,
    apply_pepper_removal,
    apply_max,
    apply_salt_removal,
    apply_sobelx,
    apply_sobely,
    apply_sobel,
    apply_laplacian,
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMAGE_STORE = {}


def create_app() -> Flask:
    app = Flask(__name__)
    # Simple dev secret; replace for production use.
    app.config["SECRET_KEY"] = "change-me"

    @app.route("/", methods=["GET"])
    def index():
        return render_template(
            "index.html",
            original_image=_get_images().get("original"),
            processed_image=_get_images().get("processed"),
        )

    @app.route("/upload", methods=["POST"])
    def upload():
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please choose an image to upload.")
            return redirect(url_for("index"))

        if not _is_allowed(file.filename):
            flash("Unsupported file type. Use PNG, JPG, JPEG, or WEBP.")
            return redirect(url_for("index"))

        image = _file_to_image(file)
        if image is None:
            flash("We could not read that image. Try a different file.")
            return redirect(url_for("index"))

        key = _ensure_image_key()
        IMAGE_STORE[key] = {
            "original": _encode_image(image),
            "processed": None,
        }
        flash("Image uploaded. Choose a filter to see the result.")
        return redirect(url_for("index"))

    @app.route("/filter", methods=["POST"])
    def apply_filter():
        filter_name = request.form.get("filter")
        images = _get_images()
        # Stack filters: use processed if present, otherwise original
        encoded_image = images.get("processed") or images.get("original")
        if not encoded_image:
            flash("Upload an image first.")
            return redirect(url_for("index"))

        image = _decode_image(encoded_image)
        filtered = _run_filter(image, filter_name)
        images["processed"] = _encode_image(filtered)
        flash("Filter applied.")
        return redirect(url_for("index"))

    @app.route("/reset", methods=["POST"]) 
    def reset():
        images = _get_images()
        if images:
            images["processed"] = None
            flash("Reset to original image.")
        else:
            flash("Upload an image first.")
        return redirect(url_for("index"))

    return app


def _is_allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _file_to_image(file_storage):
    data = file_storage.read()
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _decode_image(encoded: str):
    raw = base64.b64decode(encoded)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_image(image) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _run_filter(image, filter_name: str):

    if filter_name == "mean":
        return apply_mean(image)

    if filter_name == "median":
        return apply_median(image)

    if filter_name == "min":
        return apply_min(image)

    if filter_name == "pepper_removal":
        return apply_pepper_removal(image)

    if filter_name == "max":
        return apply_max(image)

    if filter_name == "salt_removal":
        return apply_salt_removal(image)

    if filter_name == "sobelx":
        return apply_sobelx(image)

    if filter_name == "sobely":
        return apply_sobely(image)

    if filter_name == "sobel":
        return apply_sobel(image)

    if filter_name == "laplacian":
        return apply_laplacian(image)

    # Default: a gentle glow effect.
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=8)
    return cv2.addWeighted(image, 0.7, blur, 0.3, 0)


def _ensure_image_key() -> str:
    key = session.get("image_key")
    if not key:
        key = secrets.token_urlsafe(16)
        session["image_key"] = key
    return key


def _get_images() -> dict:
    key = session.get("image_key")
    if not key:
        return {}
    return IMAGE_STORE.get(key, {})


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 9115))
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(debug=True, host=host, port=port)
