import argparse
import json
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import requests

PIPELINE_DIR = "/home/zb/Documents/active_pipeline"
sys.path.insert(0, PIPELINE_DIR)

from smoothie_cv.config import Config
from smoothie_cv.detection import detect_container
from smoothie_cv.pipelines.blend import BlendPipeline
from smoothie_cv.pipelines.spill import SpillPipeline

# ========== CONFIG ==========
ENDPOINT     = "aqvzibsexe75f-ats.iot.us-east-1.amazonaws.com"
CLIENT_ID    = "jetson-nano-smoothie"
PATH_TO_CERT = "./cert/af7f8bea61b8b9ce4d4d807fc4708e4d10b034345b9b793df0a5c758c4d4a739-certificate.pem.crt"
PATH_TO_KEY  = "./cert/af7f8bea61b8b9ce4d4d807fc4708e4d10b034345b9b793df0a5c758c4d4a739-private.pem.key"
PATH_TO_ROOT = "./cert/AmazonRootCA1.pem"
TOPIC_SUB    = "service/ml/smoothie/request"
TOPIC_PUB    = "service/ml/smoothie/result"
RESULTS_DIR  = "./results"

# concurrency controls — three model passes per image (container/chunk/spill),
# heavier than the single-model classifier, so keep this at 1 on the Nano.
MAX_WORKERS  = 1
TASK_TIMEOUT = 120
QUEUE_SIZE   = 20

smoothie_config = Config(
    yolo_weights=f"{PIPELINE_DIR}/checkpoints/yolo_standard_seg.pt",
    chunk_weights=f"{PIPELINE_DIR}/checkpoints/yolo_chunk_seg.pt",
    spill_weights=f"{PIPELINE_DIR}/checkpoints/yolo_spill_seg.pt",
)


def analyze_image(image_path: str) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    roi_mask, _bbox = detect_container(image, smoothie_config)
    roi_area = int((roi_mask > 0).sum())

    blend_result = BlendPipeline(smoothie_config).analyze(image, roi_mask)
    n_chunks = blend_result.metadata.get("n_chunks", 0)
    chunk_area_ratio = round(1.0 - blend_result.blend_score, 4) if roi_area else 0.0

    spill_result = SpillPipeline(smoothie_config).analyze(image)
    spill_area_ratio = round(spill_result.spill_area_px / roi_area, 4) if roi_area else 0.0

    return {
        "chunks": {
            "detected": n_chunks > 0,
            "count": n_chunks,
            "area_ratio": chunk_area_ratio,
        },
        "spill": {
            "detected": spill_result.spill_detected,
            "area_ratio": spill_area_ratio,
        },
    }


def process_image(payload: str, mqtt_connection=None):
    try:
        data = json.loads(payload)
        image_url = data.get("image_url")
        uuid = data.get("uuid")
        if not image_url:
            print("⚠️ No image_url in payload.")
            return

        print(f"📥 Fetching image: {image_url}")
        tmp_path = "/tmp/infer_smoothie.jpg"
        with requests.get(image_url, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)

        analysis = analyze_image(tmp_path)

        result_msg_dict = {
            "uuid": uuid,
            "image_url": image_url,
            **analysis,
            "timestamp": int(time.time()),
        }
        result_msg = json.dumps(result_msg_dict)

        Path(RESULTS_DIR).mkdir(exist_ok=True)
        out_path = Path(RESULTS_DIR) / f"{uuid or int(time.time())}.json"
        out_path.write_text(json.dumps(result_msg_dict, indent=2))
        print(f"📝 Wrote {out_path}")

        if mqtt_connection is not None:
            from awscrt import mqtt
            mqtt_connection.publish(topic=TOPIC_PUB, payload=result_msg, qos=mqtt.QoS.AT_LEAST_ONCE)
            print(f"📤 Published result to {TOPIC_PUB}")

        print(result_msg)

    except Exception as e:
        print(f"❌ Error processing image: {e}")


def run_mqtt():
    from awscrt import mqtt
    from awsiot import mqtt_connection_builder

    mqtt_connection = mqtt_connection_builder.mtls_from_path(
        endpoint=ENDPOINT,
        cert_filepath=PATH_TO_CERT,
        pri_key_filepath=PATH_TO_KEY,
        ca_filepath=PATH_TO_ROOT,
        client_id=CLIENT_ID,
        clean_session=False,
        keep_alive_secs=60,
    )
    print(f"Connecting to AWS IoT Core ({ENDPOINT})...")
    mqtt_connection.connect().result()
    print("Connected to AWS IoT Core ✅")

    task_queue = queue.Queue(maxsize=QUEUE_SIZE)
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def worker_loop():
        while True:
            payload = task_queue.get()
            if payload is None:
                break
            future = executor.submit(process_image, payload, mqtt_connection)
            try:
                future.result(timeout=TASK_TIMEOUT)
            except Exception as e:
                print(f"❌ Worker error: {e}")
            finally:
                task_queue.task_done()

    for _ in range(MAX_WORKERS):
        threading.Thread(target=worker_loop, daemon=True).start()

    def handle_message(topic, payload, dup, qos, retain, **kwargs):
        try:
            if task_queue.full():
                print("⚠️ Queue full, dropping message.")
                return
            task_queue.put(payload)
            print(f"🧩 Queued new task (size={task_queue.qsize()})")
        except Exception as e:
            print(f"❌ Error queuing message: {e}")

    print(f"Subscribing to topic: {TOPIC_SUB}")
    subscribe_future, packet_id = mqtt_connection.subscribe(
        topic=TOPIC_SUB,
        qos=mqtt.QoS.AT_LEAST_ONCE,
        callback=handle_message
    )
    subscribe_future.result()
    print("✅ Listening for smoothie analysis requests...")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("Stopping...")
        for _ in range(MAX_WORKERS):
            task_queue.put(None)
        executor.shutdown(wait=True)
        mqtt_connection.disconnect().result()
        print("Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoothie chunk/spill MQTT pipeline")
    parser.add_argument("--test", help="Run locally on a single image, no MQTT, just print/write the JSON.")
    args = parser.parse_args()

    if args.test:
        payload = json.dumps({"uuid": "local-test", "image_url": None})
        Path(RESULTS_DIR).mkdir(exist_ok=True)
        analysis = analyze_image(args.test)
        result_msg_dict = {
            "uuid": "local-test",
            "image_url": args.test,
            **analysis,
            "timestamp": int(time.time()),
        }
        out_path = Path(RESULTS_DIR) / "local-test.json"
        out_path.write_text(json.dumps(result_msg_dict, indent=2))
        print(json.dumps(result_msg_dict, indent=2))
        print(f"📝 Wrote {out_path}")
    else:
        run_mqtt()
