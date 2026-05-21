from flask import Flask, render_template, request, Response
from deepface import DeepFace
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)


@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    if request.method == "POST":

        file = request.files["image"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        analysis = DeepFace.analyze(
            img_path=path,
            actions=["age", "gender", "emotion"],
            enforce_detection=False
        )

        result = {
            "age": analysis[0]["age"],
            "gender": analysis[0]["dominant_gender"],
            "emotion": analysis[0]["dominant_emotion"],
            "image": path
        }

    return render_template("index.html", result=result)


def generate():

    while True:
        success, frame = camera.read()
        if not success:
            break

        try:
            analysis = DeepFace.analyze(
                frame,
                actions=["age", "gender", "emotion"],
                enforce_detection=False
            )

            text = f"{analysis[0]['age']} | {analysis[0]['dominant_gender']} | {analysis[0]['dominant_emotion']}"

            cv2.putText(frame, text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,255,0), 2)

        except:
            pass

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/video")
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)