"""Creates artifacts_dummy/ : a FAKE ONNX model with the real API contract (random predictions!).
Lets the Spring Boot developer integrate before the real model exists.
   python tests/make_dummy_model.py   then   MODEL_DIR=artifacts_dummy uvicorn api.main:app --port 8000"""
import json, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cropdx.constants import DISCLAIMER
from cropdx.labels import PLANTVILLAGE_38, parse_class_name

out = Path(__file__).resolve().parents[1] / "artifacts_dummy"; out.mkdir(exist_ok=True)
K = len(PLANTVILLAGE_38); rng = np.random.default_rng(0)
W = (rng.normal(0, 1, (3, K)) * 25).astype(np.float32); B = rng.normal(0, 1, K).astype(np.float32)
nodes = [helper.make_node("ReduceMean", ["input"], ["m"], axes=[2, 3], keepdims=0),
         helper.make_node("MatMul", ["m", "W"], ["mm"]), helper.make_node("Add", ["mm", "B"], ["logits"])]
g = helper.make_graph(nodes, "dummy", [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 224, 224])],
                      [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", K])],
                      [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B")])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
onnx.save(m, str(out / "model.onnx"))
(out / "labels.json").write_text(json.dumps([{"index": i, **parse_class_name(c)} for i, c in enumerate(PLANTVILLAGE_38)], indent=2))
(out / "model_meta.json").write_text(json.dumps({"modelVersion": "0.0.0-DUMMY", "architecture": "dummy", "imageSize": 224,
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "temperature": 1.0, "confidenceThreshold": 0.8,
    "numClasses": K, "testAccuracy": 0.0, "testMacroF1": 0.0, "trainedDataset": "NONE (dummy, random output)",
    "exportedAt": "n/a", "disclaimer": DISCLAIMER}, indent=2))
print("dummy model written to", out)
