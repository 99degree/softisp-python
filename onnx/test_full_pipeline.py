import os, time, threading, queue, logging, statistics
import numpy as np
import onnxruntime as ort
import argparse

# -------------------------------
# Config
# -------------------------------
Q_MAX = 8
logging.basicConfig(
    level=logging.DEBUG,  # crank up logging
    format="%(asctime)s [%(levelname)s] %(threadName)s %(message)s"
)

# Shared timing stats
thread_stats = {
    "camera": [],
    "algos": [],
    "coord": [],
    "isp": []
}

# Shared counters for visibility
thread_counts = {
    "camera": 0,
    "algos": 0,
    "coord": 0,
    "isp": 0
}

def parse_args():
    parser = argparse.ArgumentParser(description="Harness test for ONNX pipeline")
    parser.add_argument("--model_dir", required=True,
                        help="Folder containing test_algo.onnx, algo.onnx, applier.onnx")
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"],
                        help="Execution providers list (default CPUExecutionProvider)")
    parser.add_argument("--width", type=int, default=64,
                        help="Synthetic frame width/height")
    parser.add_argument("--mode", choices=["synthetic","onnx"], default="synthetic",
                        help="Camera mode")
    return parser.parse_args()

# -------------------------------
# Helpers
# -------------------------------
def drop_oldest_and_put(q: queue.Queue, item, qname="queue"):
    try:
        q.put(item, timeout=0.02)
        logging.debug(f"[{qname}] put ok (size={q.qsize()})")
    except queue.Full:
        logging.warning(f"[{qname}] full; dropping oldest")
        try:
            q.get_nowait()
            logging.debug(f"[{qname}] dropped oldest (size={q.qsize()})")
        except queue.Empty:
            logging.debug(f"[{qname}] empty when dropping")
        q.put(item)
        logging.debug(f"[{qname}] put after drop (size={q.qsize()})")

def debug_session_io(sess: ort.InferenceSession, outs=None, title=""):
    in_info = sess.get_inputs()
    out_info = sess.get_outputs()
    out_names = [o.name for o in out_info]
    if outs is None:
        logging.info(f"=== {title or 'session'} inputs ===")
        for i, inp in enumerate(in_info):
            logging.info(f"Input {i}: name={inp.name}, type={inp.type}, shape={inp.shape}")
        logging.info(f"=== {title or 'session'} outputs ===")
        for i, out in enumerate(out_info):
            logging.info(f"Output {i}: name={out.name}, type={out.type}, shape={out.shape}")
    else:
        logging.info(f"=== {title or 'session'} output values ===")
        for name, arr in zip(out_names, outs):
            logging.info(f"{name}: shape={getattr(arr,'shape',None)}, dtype={getattr(arr,'dtype',None)}")

def map_outs_to_named_dict(output_meta, outs):
    if output_meta is None:
       return None;
    d = {meta.name: arr for meta, arr in zip(output_meta, outs)}
    logging.debug(f"[map_outs] keys={list(d.keys())}")
    return d

def filter_feed_for_inputs(feed_dict, input_meta):
    if feed_dict is None:
       return None
    names = [meta.name for meta in input_meta]
    filtered = {name: feed_dict[name] for name in names if name in feed_dict}
    logging.debug(f"[filter_feed] in={len(feed_dict)} kept={list(filtered.keys())}")
    return filtered

def save_ppm(filename, pixels, width, height, maxval=255):
    with open(filename, "wb") as f:
        f.write(f"P6\n{width} {height}\n{maxval}\n".encode())
        for row in pixels:
            for (r,g,b) in row:
                f.write(bytes([r,g,b]))
'''
# Example: img is [h,w,3] with values 0..255
save_ppm("out.ppm", img.tolist(), img.shape[1], img.shape[0])
'''
# -------------------------------
# Synthetic generator
# -------------------------------
def generate_bar_bayer(width=64, frame_id=0,
                       wb_gains=(1.0, 1.0, 1.0),
                       ccm=np.eye(3, dtype=np.float32),
                       offset=0.1):
    logging.debug(f"[gen] width={width} frame_id={frame_id} wb={wb_gains} offset={offset}")
    height = width
    bayer = np.zeros((height, width), dtype=np.int16)
    bar_width = max(1, width // 6)
    colors = [(1023, 0, 0), (0, 1023, 0), (0, 0, 1023)]
    shift = frame_id % bar_width
    repeats = (width // (3 * bar_width)) + 2
    for i, (r_val, g_val, b_val) in enumerate(colors * repeats):
        start = i * bar_width + shift
        end = min(start + bar_width, width)
        if start >= width:
            break
        bayer[0::2, start:end:2] = int(r_val * wb_gains[0])
        bayer[0::2, start+1:end:2] = int(g_val * wb_gains[1])
        bayer[1::2, start:end:2] = int(g_val * wb_gains[1])
        bayer[1::2, start+1:end:2] = int(b_val * wb_gains[2])
    out = {
        "image_desc.input.image": bayer.astype(np.int16),
        "image_desc.input.width": np.array([width], dtype=np.int64),
        "image_desc.input.frame_id": np.array([frame_id], dtype=np.int64),
        "blacklevel.offset": np.array([offset], dtype=np.float32),
        "image_desc.input.image.function": bayer.astype(np.int16),
        "image_desc.input.width.function": np.array([width], dtype=np.int64),
        "image_desc.input.frame_id.function": np.array([frame_id], dtype=np.int64),
        "blacklevel.offset.function": np.array([offset], dtype=np.float32),
    }
    logging.debug(f"[gen] out keys={list(out.keys())}")
    return out

# -------------------------------
# Threads
# -------------------------------
def camera_thread(cam_to_algo_q, cam_to_coord_q, isp_in_q, stop_event,
                  mode="synthetic", test_algo_sess=None,
                  width=64, wb_gains=(1.0,1.0,1.0),
                  ccm=np.eye(3, dtype=np.float32)):
    logging.info("[Camera] start")
    frame_id = 0
    wb = list(wb_gains)
    offset = 0.1

    # Simple backpressure parameters
    isp_threshold = max(1, Q_MAX - 1)   # if isp_in_q reaches this, camera will back off
    sleep_on_full = 0.01                # sleep duration when ISP queue is full (seconds)

    while not stop_event.is_set():
        # If ISP queue is saturated, back off at source
        if isp_in_q.qsize() >= isp_threshold:
            logging.debug("[Camera] isp_in_q full (size=%d); sleeping %.3fs", isp_in_q.qsize(), sleep_on_full)
            time.sleep(sleep_on_full)
            continue

        if cam_to_coord_q.qsize() >= isp_threshold:
            logging.debug("[Camera] cam_to_coord_q full (size=%d); sleeping %.3fs", cam_to_coord_q.qsize(), sleep_on_full)
            time.sleep(sleep_on_full)
            continue

        start = time.perf_counter()
        try:
            if mode == "onnx" and test_algo_sess is not None:
                logging.debug("[Camera] running test_algo.onnx")
                outs = test_algo_sess.run(None, {})
                drop_oldest_and_put(cam_to_algo_q, outs, qname="cam→algo")
                drop_oldest_and_put(cam_to_coord_q, outs, qname="cam→coord")
            else:
                wb = [min(max(g+0.001,0.8),1.2) for g in wb]
                offset += 0.001
                if offset > 0.2:
                    offset = 0.1
                outs_dict = generate_bar_bayer(width=width, frame_id=frame_id,
                                               wb_gains=tuple(wb), ccm=ccm,
                                               offset=offset)
                drop_oldest_and_put(cam_to_algo_q, outs_dict, qname="cam→algo")
                drop_oldest_and_put(cam_to_coord_q, outs_dict, qname="cam→coord")
        except Exception as e:
            logging.exception(f"[Camera][Frame {frame_id}] failed: {e}")

        frame_id += 1
        time.sleep(0.01)
        end = time.perf_counter()
        thread_stats["camera"].append(end-start)
        thread_counts["camera"] += 1

    logging.info("[Camera] stop")

def algos_thread(cam_to_algo_q, algo_to_coord_q, stop_event,
                 test_algo_sess, algo_sess):
    logging.info("[Algos] start")
    if test_algo_sess is not None:
        test_out_meta = test_algo_sess.get_outputs()
    algo_in_meta = algo_sess.get_inputs()
    while not stop_event.is_set():
        start = time.perf_counter()
        try:
            msg = cam_to_algo_q.get(timeout=0.1)
            logging.debug(f"[Algos] got msg type={type(msg).__name__}")
        except queue.Empty:
            continue
        if msg is None:
            logging.debug("[Algos] msg is None; skip")
            continue
        try:
            if isinstance(msg, dict):
                feed = filter_feed_for_inputs(msg, algo_in_meta)
            else:
                feed_a = map_outs_to_named_dict(test_out_meta, msg)
                feed = filter_feed_for_inputs(feed_a, algo_in_meta)
            logging.debug(f"[Algos] feed keys={list(feed.keys())}")
            outs = algo_sess.run(None, feed)
            drop_oldest_and_put(algo_to_coord_q, outs, qname="algo→coord")
        except Exception as e:
            logging.exception(f"[Algos] failed: {e}")
        end = time.perf_counter()
        thread_stats["algos"].append(end-start)
        thread_counts["algos"] += 1
    logging.info("[Algos] stop")

def coordinator_thread(cam_to_coord_q, algo_to_coord_q, isp_in_q,
                       stop_event, test_algo_sess, algo_sess):
    logging.info("[Coord] start")
    if test_algo_sess is not None:
        test_out_meta = test_algo_sess.get_outputs()
    algo_out_meta = algo_sess.get_outputs()
    while not stop_event.is_set():
        start = time.perf_counter()
        latest_camera = None
        latest_algo = None
        try:
            while True:
                latest_camera = cam_to_coord_q.get_nowait()
        except queue.Empty:
            pass
        try:
            while True:
                latest_algo = algo_to_coord_q.get_nowait()
        except queue.Empty:
            pass
        if latest_camera is None or latest_algo is None:
            logging.debug("[Coord] waiting for both camera+algo")
            time.sleep(0.005)
            end = time.perf_counter()
            thread_stats["coord"].append(end-start)
            thread_counts["coord"] += 1
            continue
        feed_a = latest_camera if isinstance(latest_camera, dict) else map_outs_to_named_dict(test_out_meta, latest_camera)
        feed_b = map_outs_to_named_dict(algo_out_meta, latest_algo)
        feed_c = {**feed_a, **feed_b}
        logging.debug(f"[Coord] merged keys={list(feed_c.keys())}")
        drop_oldest_and_put(isp_in_q, feed_c, qname="coord→isp")
        time.sleep(0.001)
        end = time.perf_counter()
        thread_stats["coord"].append(end-start)
        thread_counts["coord"] += 1
    logging.info("[Coord] stop")

def isp_thread(isp_in_q, stop_event, isp_sess):
    logging.info("[ISP] start")
    isp_in_meta = isp_sess.get_inputs()
    frame_done = 0
    while not stop_event.is_set():
        start = time.perf_counter()
        try:
            big_feed = isp_in_q.get(timeout=0.1)
            logging.debug(f"[ISP] got feed type={type(big_feed).__name__}")
        except queue.Empty:
            continue
        if big_feed is None:
            logging.debug("[ISP] feed None; skip")
            continue
        try:
            feed_a = {meta.name: big_feed[meta.name] for meta in isp_in_meta if meta.name in big_feed}
            logging.debug(f"[ISP] filtered keys={list(feed_a.keys())}")
            outputs = isp_sess.run(None, feed_a)
            logging.debug(f"[ISP] outputs count={len(outputs)}")
        except Exception as e:
            logging.exception(f"[ISP][Frame {frame_done}] failed: {e}")
        frame_done += 1
        end = time.perf_counter()
        thread_stats["isp"].append(end-start)
        thread_counts["isp"] += 1
    logging.info("[ISP] stop")

# -------------------------------
# Main
# -------------------------------
def main():
    logging.info("[Main] parse args")
    args = parse_args()
    providers = args.providers

    MODEL_DIR = args.model_dir
    TEST_ALGO_ONNX = os.path.join(MODEL_DIR, "test_algo.onnx")
    ALGO_ONNX      = os.path.join(MODEL_DIR, "algo.onnx")
    ISP_ONNX       = os.path.join(MODEL_DIR, "applier.onnx")
    '''
    # Sessions with robust error logging
    try:
        logging.info(f"[Main] load test_algo: {TEST_ALGO_ONNX}")
        test_algo_sess = ort.InferenceSession(TEST_ALGO_ONNX, providers=providers)
        debug_session_io(test_algo_sess, title="test_algo.onnx")
    except Exception as e:
        logging.exception(f"[Main] failed to create test_algo session: {e}")
        return
    '''
    test_algo_sess = None

    try:
        logging.info(f"[Main] load algo: {ALGO_ONNX}")
        algo_sess = ort.InferenceSession(ALGO_ONNX, providers=providers)
        debug_session_io(algo_sess, title="algo.onnx")
    except Exception as e:
        logging.exception(f"[Main] failed to create algo session: {e}")
        return

    try:
        logging.info(f"[Main] load ISP: {ISP_ONNX}")
        isp_sess = ort.InferenceSession(ISP_ONNX, providers=providers)
        debug_session_io(isp_sess, title="applier.onnx")
    except Exception as e:
        logging.exception(f"[Main] failed to create ISP session: {e}")
        return

    # Queues
    cam_to_algo_q  = queue.Queue(maxsize=Q_MAX)
    cam_to_coord_q = queue.Queue(maxsize=Q_MAX)
    algo_to_coord_q= queue.Queue(maxsize=Q_MAX)
    isp_in_q       = queue.Queue(maxsize=Q_MAX)

    stop_event = threading.Event()

    # Threads (non‑daemon so they keep process alive)
    threads = [
        threading.Thread(name="Camera", target=camera_thread,
                         args=(cam_to_algo_q, cam_to_coord_q, isp_in_q, stop_event,
                               args.mode, test_algo_sess, args.width)),
        threading.Thread(name="Algos", target=algos_thread,
                         args=(cam_to_algo_q, algo_to_coord_q, stop_event,
                               test_algo_sess, algo_sess)),
        threading.Thread(name="Coord", target=coordinator_thread,
                         args=(cam_to_coord_q, algo_to_coord_q, isp_in_q,
                               stop_event, test_algo_sess, algo_sess)),
        threading.Thread(name="ISP", target=isp_thread,
                         args=(isp_in_q, stop_event, isp_sess)),
    ]
    for t in threads:
        logging.info(f"[Main] starting thread {t.name}")
        t.start()

    try:
        # Heartbeat so you know it's alive
        while not stop_event.is_set():
            logging.info(f"[Main] alive | counts: "
                         f"camera={thread_counts['camera']} "
                         f"algos={thread_counts['algos']} "
                         f"coord={thread_counts['coord']} "
                         f"isp={thread_counts['isp']}")
            time.sleep(5.0)
    except KeyboardInterrupt:
        logging.info("[Main] Ctrl+C received, stopping...")
        stop_event.set()

    # Wait for threads to finish
    for t in threads:
        logging.info(f"[Main] joining {t.name}")
        t.join(timeout=2.0)

    # Final timing summary
    print("\n=== Timing summary ===")
    for name, durs in thread_stats.items():
        if durs:
            print(f"{name}: min={min(durs):.6f}s "
                  f"max={max(durs):.6f}s "
                  f"mean={statistics.mean(durs):.6f}s "
                  f"count={len(durs)}")
        else:
            print(f"{name}: no iterations recorded")

if __name__ == "__main__":
    main()
