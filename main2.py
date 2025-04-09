import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

def tile_image(image, tile_size, overlap):
    """Splits the image into overlapping tiles.

    Returns a list of tuples (tile, offset_x, offset_y).
    """
    h, w = image.shape[:2]
    tiles = []
    stride = tile_size - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Ensure we don't go out of bounds
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            tile = image[y:y_end, x:x_end]
            tiles.append((tile, x, y))
    return tiles

def run_inference_on_tiles(frame, model, tile_size=640, overlap=100, conf=0.05, iou=0.1):
    """
    Runs YOLOv8 model on tiles of the input frame.
    Returns combined detections with coordinates corresponding to the original frame.
    """
    detections = []
    tiles = tile_image(frame, tile_size, overlap)
    for tile, offset_x, offset_y in tiles:
        results = model(tile, conf=conf, iou=iou)
        # YOLOv8 returns a list; use the first result (assuming a single image was passed)
        for det in results[0].boxes.data.cpu().numpy():
            # Each detection: [x1, y1, x2, y2, conf, cls]
            x1, y1, x2, y2, score, cls_id = det
            # Adjust coordinates back to the full image
            x1_full = x1 + offset_x
            y1_full = y1 + offset_y
            x2_full = x2 + offset_x
            y2_full = y2 + offset_y
            detections.append([x1_full, y1_full, x2_full, y2_full, score, cls_id])
    return detections

def main():
    # Initialize ZED camera with high resolution
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.camera_resolution = sl.RESOLUTION.HD2K  # Use a higher resolution
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open failed:", status)
        exit(1)

    # Enable positional tracking if needed
    zed.enable_positional_tracking(sl.PositionalTrackingParameters())

    # Retrieve camera resolution and create image object
    cam_info = zed.get_camera_information()
    resolution = cam_info.camera_configuration.resolution
    image_left = sl.Mat()

    # Load YOLOv8 model (use custom weights if available)
    model = YOLO("yolov8n.pt")

    window_name = "Tiled YOLOv8 Small Particle Detection"
    while True:
        err = zed.grab()
        if err != sl.ERROR_CODE.SUCCESS:
            print("Failed to grab frame")
            break

        # Retrieve the left image from the ZED camera
        zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, resolution)
        frame = image_left.get_data()  # BGRA format
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Run inference over image tiles for small object detection from a distance
        detections = run_inference_on_tiles(frame_bgr, model, tile_size=640, overlap=100,
                                            conf=0.05, iou=0.1)

        # Draw detections on the full frame
        annotated_frame = frame_bgr.copy()
        for det in detections:
            x1, y1, x2, y2, score, cls_id = det
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{int(cls_id)}: {score:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow(window_name, annotated_frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
