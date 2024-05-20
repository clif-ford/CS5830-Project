from fastapi import FastAPI, File, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import start_http_server, Counter, Gauge
import psutil
import pickle

from components import load_model, predict_digit, read_imagefile, format_image
from time import time




# Initialize Prometheus metrics
num_requests = Counter('num_requests', 'Number of requests received', ['method', 'endpoint', 'ip_address'])
processing_time_per_char = Gauge('processing_time_per_char', 'Processing time per character in microseconds', ['method', 'endpoint'])
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
network_io_sent = Counter('network_io_sent_bytes_total', 'Total number of bytes sent via network')
network_io_received = Counter('network_io_received_bytes_total', 'Total number of bytes received via network')
processing_time = Gauge('processing_time', 'Processing time in microseconds', ['method', 'endpoint'])

app = FastAPI()

model = load_model('handwritten_character_model.h5')
with open('onehot_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

@app.get("/")
async def home():
    return "App is functional."


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)) -> dict:
    """
    Passes an uploaded image to the model to evaluate and guess what the number on the image is

    Args:
        file (UploadFile, optional): The image file uploaded by the user to be evaluated by the model. Defaults to File(...).

    Returns:
        dict: a dictionary which simply returns the digit guessed by the model
    """
    try:
        start_time = time()
        img = read_imagefile(await file.read())
        pred_inp, img_size = format_image(img)
        end_time = time()
        processing_time_val = (end_time - start_time) * 1000
        processing_time_per_char_value = (processing_time_val / img_size) * 1000
        digit = predict_digit(model=model, data_point=pred_inp)
        digit = encoder.inverse_transform(digit)
        digit = digit[0][0]
        
        processing_time.labels(method="POST", endpoint="/predict").set(processing_time_val)
        processing_time_per_char.labels(method="POST", endpoint="/predict").set(processing_time_per_char_value)

        memory_usage.set(psutil.virtual_memory().used)
        cpu_usage.set(psutil.cpu_percent())
        net_io = psutil.net_io_counters()
        network_io_received.inc(net_io.bytes_recv)
        network_io_sent.inc(net_io.bytes_sent)
    except:
        return "Error reading file"
    
    return {"character" : digit}

start_http_server(18000)

Instrumentator().instrument(app).expose(app)
