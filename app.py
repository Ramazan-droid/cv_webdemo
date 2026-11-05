import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from torchvision import models

app = Flask(__name__)


CLASSES = ["fist", "peace", "stop"]

MODEL_PATHS = {
    "baseline": "models/trslearningmodel.pth",
    "sota": "models/SOTA_model_weights.pth",
}
# -----------------------------------

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
device = torch.device("cpu")


class TransferLearningModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # weights=None: we load our own weights, no ImageNet download needed
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


_cache = {}


def get_model(name):
    # lazy-load so the free instance starts fast and uses less RAM
    if name not in _cache:
        m = TransferLearningModel(len(CLASSES))
        m.load_state_dict(torch.load(MODEL_PATHS[name], map_location=device))
        m.eval()
        _cache[name] = m
    return _cache[name]


def preprocess(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)

def simple_cam(model, x):
    acts = []

    def hook_fn(module, input, output):
        acts.append(output.detach())

    handle = model.model.layer4.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            logits = model(x)

        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())

        cam = acts[0][0].mean(dim=0).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return probs, idx, cam

    finally:
        handle.remove()
# def predict_with_gradcam(model, x):
#     acts = []
#     hook = model.model.layer4.register_forward_hook(lambda m, i, o: acts.append(o))
#     # backbone is frozen, so input must require grad for the graph to exist
#     x.requires_grad_(True)
#     logits = model(x)
#     hook.remove()

#     probs = F.softmax(logits, dim=1)[0].detach().numpy()
#     idx = int(probs.argmax())

#     grads = torch.autograd.grad(logits[0, idx], acts[0])[0]
#     weights = grads.mean(dim=(2, 3), keepdim=True)
#     cam = F.relu((weights * acts[0]).sum(dim=1))[0].detach().numpy()
#     cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
#     return probs, idx, cam


@app.route("/")
@app.route("/baseline")
def baseline():
    return render_template("index.html", model="baseline", title="Baseline (transfer learning)")


@app.route("/sota")
def sota():
    return render_template("index.html", model="sota", title="SOTA (trained with synthetic data)")


@app.route("/predict/<name>", methods=["POST"])
def predict(name):
    if name not in MODEL_PATHS:
        return jsonify(error="unknown model"), 404

    b64 = request.json["image"].split(",")[1]
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)

    probs, idx, cam = simple_cam(get_model(name), preprocess(frame))

    heat = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.55, heat, 0.45, 0)
    _, buf = cv2.imencode(".jpg", overlay)

    return jsonify(
        label=CLASSES[idx],
        confidence=float(probs[idx]),
        probs={c: float(p) for c, p in zip(CLASSES, probs)},
        heatmap="data:image/jpeg;base64," + base64.b64encode(buf).decode(),
    )


if __name__ == "__main__":
    app.run(debug=True)