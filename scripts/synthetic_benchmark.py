import time
import numpy as np
import onnxruntime as ort

def benchmark_tflite(model_path, shape, is_tflite=True):
    if is_tflite:
        try:
            from ai_edge_litert.interpreter import Interpreter
            interp = Interpreter(model_path=model_path, num_threads=4)
            interp.allocate_tensors()
            idx = interp.get_input_details()[0]['index']
            for _ in range(5):
                interp.set_tensor(idx, np.zeros(shape, dtype=np.float32))
                interp.invoke()
            t0 = time.perf_counter()
            for _ in range(30):
                interp.set_tensor(idx, np.zeros(shape, dtype=np.float32))
                interp.invoke()
            t1 = time.perf_counter()
            return (t1 - t0) * 1000 / 30
        except Exception as e:
            return str(e)
    else:
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            sess = ort.InferenceSession(model_path, so)
            idx = sess.get_inputs()[0].name
            for _ in range(5):
                sess.run(None, {idx: np.zeros(shape, dtype=np.float32)})
            t0 = time.perf_counter()
            for _ in range(30):
                sess.run(None, {idx: np.zeros(shape, dtype=np.float32)})
            t1 = time.perf_counter()
            return (t1 - t0) * 1000 / 30
        except Exception as e:
            return str(e)

print(f"YOLOv8 ONNX (CPU): {benchmark_tflite('yolov8n-seg.onnx', (1, 3, 640, 640), False)}")
print(f"YOLOv8 TFLite (CPU XNNPACK): {benchmark_tflite('models/yolo_tflite/yolov8n-seg_float32.tflite', (1, 3, 640, 640), True)}")
print(f"DINOv2 ONNX (CPU): {benchmark_tflite('dinov2_vits14.onnx', (1, 3, 224, 224), False)}")
print(f"DINOv2 TFLite (CPU XNNPACK): {benchmark_tflite('models/dinov2_tflite/dinov2_vits14_float32.tflite', (1, 3, 224, 224), True)}")
