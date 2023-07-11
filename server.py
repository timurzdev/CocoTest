import io

from PIL import Image

from flask import Flask, jsonify, request
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((500, 500)),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


def get_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(91)
checkpoint = torch.load("./model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected for uploading"}), 400
        if file:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            image = transform(image)
            image = image.unsqueeze(0)
            image = image.to(device)

            with torch.no_grad():
                prediction = model(image)

            boxes = prediction[0]["boxes"]
            scores = prediction[0]["scores"]
            labels = prediction[0]["labels"]

            score_threshold = 0.1
            indices = [i for i, score in enumerate(scores) if score > score_threshold]
            boxes = boxes[indices]
            scores = scores[indices]
            labels = labels[indices]
            keep = nms(boxes, scores, 0.5)

            prediction[0]["boxes"] = boxes[keep]
            prediction[0]["scores"] = scores[keep]
            prediction[0]["labels"] = labels[keep]
            for key, value in prediction[0].items():
                prediction[0][key] = value.tolist()
            return jsonify(prediction[0])


if __name__ == "__main__":
    app.run(debug=False, port=7777)
