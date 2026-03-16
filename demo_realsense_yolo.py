import time
import cv2
import numpy as np
import tkinter as tk
import pyrealsense2 as rs
from ultralytics import YOLO

MODEL_NAME = "yolo26n.pt"
CONF_THRESHOLD = 0.35
IMG_SIZE = 640
WINDOW_NAME = "RealSense + YOLO"

# Mettre True pour limiter les objets affichés pendant l'atelier
INTEREST_ONLY = False
INTEREST_CLASSES = {
    "person",
    "apple",
    "banana",
    "orange",
    "bottle",
    "cup",
    "cell phone",
    "book",
    "mouse",
    "keyboard",
    "scissors",
    "teddy bear",
}

# Traduction FR complète des 80 classes COCO
LABELS_FR = {
    "person": "personne",
    "bicycle": "velo",
    "car": "voiture",
    "motorcycle": "moto",
    "airplane": "avion",
    "bus": "bus",
    "train": "train",
    "truck": "camion",
    "boat": "bateau",
    "traffic light": "feu tricolore",
    "fire hydrant": "borne incendie",
    "stop sign": "panneau stop",
    "parking meter": "parcmetre",
    "bench": "banc",
    "bird": "oiseau",
    "cat": "chat",
    "dog": "chien",
    "horse": "cheval",
    "sheep": "mouton",
    "cow": "vache",
    "elephant": "elephant",
    "bear": "ours",
    "zebra": "zebre",
    "giraffe": "girafe",
    "backpack": "sac a dos",
    "umbrella": "parapluie",
    "handbag": "sac a main",
    "tie": "cravate",
    "suitcase": "valise",
    "frisbee": "frisbee",
    "skis": "skis",
    "snowboard": "snowboard",
    "sports ball": "ballon",
    "kite": "cerf volant",
    "baseball bat": "batte de baseball",
    "baseball glove": "gant de baseball",
    "skateboard": "skateboard",
    "surfboard": "planche de surf",
    "tennis racket": "raquette de tennis",
    "bottle": "bouteille",
    "wine glass": "verre a vin",
    "cup": "tasse",
    "fork": "fourchette",
    "knife": "couteau",
    "spoon": "cuillere",
    "bowl": "bol",
    "banana": "banane",
    "apple": "pomme",
    "sandwich": "sandwich",
    "orange": "orange",
    "broccoli": "brocoli",
    "carrot": "carotte",
    "hot dog": "hot dog",
    "pizza": "pizza",
    "donut": "donut",
    "cake": "gateau",
    "chair": "chaise",
    "couch": "canape",
    "potted plant": "plante en pot",
    "bed": "lit",
    "dining table": "table",
    "toilet": "toilettes",
    "tv": "ecran",
    "laptop": "ordinateur",
    "mouse": "souris",
    "remote": "telecommande",
    "keyboard": "clavier",
    "cell phone": "telephone",
    "microwave": "micro ondes",
    "oven": "four",
    "toaster": "grille pain",
    "sink": "evier",
    "refrigerator": "refrigerateur",
    "book": "livre",
    "clock": "horloge",
    "vase": "vase",
    "scissors": "ciseaux",
    "teddy bear": "peluche",
    "hair drier": "seche cheveux",
    "toothbrush": "brosse a dents",
}


def draw_label(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2

    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)

    x = max(0, min(x, img.shape[1] - w - 10))
    y = max(h + baseline + 8, min(y, img.shape[0] - 4))

    cv2.rectangle(
        img,
        (x, y - h - baseline - 8),
        (x + w + 8, y),
        (0, 255, 0),
        -1,
    )
    cv2.putText(
        img,
        text,
        (x + 4, y - 4),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def resize_and_crop_to_fill(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = max(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    crop_x = (new_w - target_w) // 2
    crop_y = (new_h - target_h) // 2

    cropped = resized[crop_y:crop_y + target_h, crop_x:crop_x + target_w]
    return cropped, scale, crop_x, crop_y


def main():
    print(f"[INFO] Chargement du modele {MODEL_NAME} ...")
    model = YOLO(MODEL_NAME)

    print("[INFO] Demarrage RealSense ...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    prev_time = time.time()
    fullscreen = True

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            h, w, _ = frame.shape

            result = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                imgsz=IMG_SIZE,
                verbose=False,
                device="cpu",
            )[0]

            display_full, scale, crop_x, crop_y = resize_and_crop_to_fill(
                frame, screen_w, screen_h
            )

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    name_en = result.names[cls_id]
                    name_fr = LABELS_FR.get(name_en, name_en)

                    if INTEREST_ONLY and name_en not in INTEREST_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    distance_m = depth_frame.get_distance(cx, cy)

                    tx1 = int(x1 * scale) - crop_x
                    ty1 = int(y1 * scale) - crop_y
                    tx2 = int(x2 * scale) - crop_x
                    ty2 = int(y2 * scale) - crop_y
                    tcx = int(cx * scale) - crop_x
                    tcy = int(cy * scale) - crop_y

                    # Si la boite est totalement hors ecran apres crop, on ignore
                    if tx2 < 0 or ty2 < 0 or tx1 >= screen_w or ty1 >= screen_h:
                        continue

                    tx1 = max(0, min(tx1, screen_w - 1))
                    ty1 = max(0, min(ty1, screen_h - 1))
                    tx2 = max(0, min(tx2, screen_w - 1))
                    ty2 = max(0, min(ty2, screen_h - 1))
                    tcx = max(0, min(tcx, screen_w - 1))
                    tcy = max(0, min(tcy, screen_h - 1))

                    cv2.rectangle(display_full, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
                    cv2.circle(display_full, (tcx, tcy), 5, (0, 255, 255), -1)

                    label = f"{name_fr} {conf:.2f}"
                    if distance_m > 0:
                        label += f" | {distance_m:.2f} m"

                    draw_label(display_full, label, tx1, ty1)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            cv2.putText(
                display_full,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display_full,
                "q ou ESC pour quitter",
                (20, screen_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, display_full)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("f"):
                fullscreen = not fullscreen
                mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)

            if key == ord("q") or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
