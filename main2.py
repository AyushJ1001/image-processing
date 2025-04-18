import cv2
import numpy as np
import pyzed.sl as sl
import math # For color distance calculation

# --- Parameters for Particle Filtering --- STICTER VALUES ---
# *** ADJUST THESE BASED ON YOUR ACTUAL PARTICLE SIZE AND SHAPE ***
MIN_PARTICLE_AREA = 15  # Increased min area example
MAX_PARTICLE_AREA = 200 # Decreased max area example
MIN_CIRCULARITY = 0.6 # Increased circularity requirement <-- REVERTED TO LESS STRICT
MAX_CIRCULARITY = 0.95 # Example: Exclude near-perfect circles (adjust as needed)
# Parameters for SimpleBlobDetector
BLOB_MIN_THRESHOLD = 10
BLOB_MAX_THRESHOLD = 250  # Increased threshold to include brighter (white) blobs
BLOB_MIN_CONVEXITY = 0.80 # Increased convexity requirement <-- REVERTED TO LESS STRICT
BLOB_MIN_INERTIA_RATIO = 0.1 # Keep relatively low unless particles are perfectly circular
# --- Parameters for Particle Matching (using Lab color space) ---
# Max *perceptual* distance in Lab color space to consider particles a match.
# Lab distances typically range up to ~100+. Smaller values mean stricter color matching.
# Good values often range from 10 to 30, depending on expected variance.
LAB_MATCH_THRESHOLD = 50.0 # Adjust this value based on observed color variance

# Removed COLOR_RANGES_HSV and DRAW_COLORS_BGR as color is now sampled

def setup_blob_detector():
    """Sets up the SimpleBlobDetector."""
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = BLOB_MIN_THRESHOLD
    params.maxThreshold = BLOB_MAX_THRESHOLD
    params.filterByArea = True
    params.minArea = MIN_PARTICLE_AREA
    params.maxArea = MAX_PARTICLE_AREA
    params.filterByCircularity = True
    params.minCircularity = MIN_CIRCULARITY
    params.maxCircularity = MAX_CIRCULARITY
    params.filterByConvexity = True
    params.minConvexity = BLOB_MIN_CONVEXITY
    params.filterByInertia = True
    params.minInertiaRatio = BLOB_MIN_INERTIA_RATIO
    detector = cv2.SimpleBlobDetector_create(params)
    return detector

def detect_particles_and_sample_color(image_bgr, detector):
    """Detects blobs and samples their BGR and Lab colors."""
    detected_list = []
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # --- Add Morphological Opening to remove small noise --- #
    kernel = np.ones((3,3), np.uint8) # 3x3 kernel, adjust size if needed
    gray_opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    # Optional: Display opened image for debugging
    # cv2.imshow("Opened Gray", gray_opened)

    # Use inverted image if particles are dark on light background
    # keypoints = detector.detect(255 - gray_opened)
    keypoints = detector.detect(gray_opened) # Detect on the cleaned image

    for kp in keypoints:
        cX = int(kp.pt[0])
        cY = int(kp.pt[1])
        if 0 <= cX < width and 0 <= cY < height:
            # Sample BGR
            color_bgr = tuple(map(int, image_bgr[cY, cX]))

            # Convert *just this pixel* BGR to Lab for matching
            # Need reshape for cvtColor: create 1x1 pixel image
            pixel_bgr = np.uint8([[color_bgr]])
            pixel_lab = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2LAB)
            color_lab = tuple(map(int, pixel_lab[0, 0])) # Extract Lab values

            # Store centroid, BGR (for drawing), and Lab (for matching)
            detected_list.append(((cX, cY), color_bgr, color_lab))
    return detected_list

def lab_color_distance(color1_lab, color2_lab):
    """Calculates Euclidean distance between two Lab colors."""
    l1, a1, b1 = color1_lab
    l2, a2, b2 = color2_lab
    return math.sqrt((l1 - l2)**2 + (a1 - a2)**2 + (b1 - b2)**2)

def match_particles(start_particles, end_particles, max_lab_dist):
    """
    Matches particles between start and end lists based on closest Lab color distance.
    Expects particle data as ((x,y), (b,g,r), (L,a,b)).
    """
    num_start = len(start_particles)
    num_end = len(end_particles)
    if num_start == 0 or num_end == 0:
        return [], start_particles, end_particles

    potential_matches = []
    for i in range(num_start):
        start_lab = start_particles[i][2] # Get Lab color tuple
        for j in range(num_end):
            end_lab = end_particles[j][2]   # Get Lab color tuple
            dist = lab_color_distance(start_lab, end_lab)
            if dist <= max_lab_dist:
                potential_matches.append((dist, i, j))

    potential_matches.sort()
    matched_pairs = []
    matched_start_indices = set()
    matched_end_indices = set()
    for dist, start_idx, end_idx in potential_matches:
        if start_idx not in matched_start_indices and end_idx not in matched_end_indices:
            # Return the full particle data tuple ((x,y), (b,g,r), (L,a,b))
            matched_pairs.append((start_particles[start_idx], end_particles[end_idx]))
            matched_start_indices.add(start_idx)
            matched_end_indices.add(end_idx)

    unmatched_start = [p for i, p in enumerate(start_particles) if i not in matched_start_indices]
    unmatched_end = [p for i, p in enumerate(end_particles) if i not in matched_end_indices]

    return matched_pairs, unmatched_start, unmatched_end

def draw_detections(image, particles_list, displacements_map=None):
    """Draws detected centroids (colored by BGR) and optional displacement vectors."""
    vis_image = image.copy()
    for particle_data in particles_list:
        centroid, color_bgr, _ = particle_data
        cx, cy = centroid
        draw_color = tuple(map(int, color_bgr))
        # Draw centroid circle
        cv2.circle(vis_image, (cx, cy), 7, draw_color, -1)
        cv2.circle(vis_image, (cx, cy), 7, (0, 0, 0), 1)

        # Draw displacement vector if available
        if displacements_map and centroid in displacements_map:
            dx, dy = displacements_map[centroid]
            end_point = (cx + dx, cy + dy)
            # Draw white arrow with black outline for visibility
            cv2.arrowedLine(vis_image, (cx, cy), end_point, (0, 0, 0), 3, tipLength=0.3)
            cv2.arrowedLine(vis_image, (cx, cy), end_point, (255, 255, 255), 1, tipLength=0.3)
            # Optional: Put text near the arrow end
            # cv2.putText(vis_image, f"({dx},{dy})", (end_point[0] + 5, end_point[1] + 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return vis_image

def main():
    blob_detector = setup_blob_detector()
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open failed: {status}")
        zed.close()
        exit(1)
    cam_info = zed.get_camera_information()
    resolution = cam_info.camera_configuration.resolution
    image_zed = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.U8_C4)
    start_image = None
    start_particles = []
    end_particles = []
    start_frame_captured = False
    start_vis_image = None
    results_calculated = False
    window_live = "ZED Live Feed"
    window_start_detections = "Start Frame Detections"
    print("Camera Initialized. Press 's' -> move particles -> 'e'. 'r' to reset, 'q' to quit.")
    print(f"Using Lab Color Matching Threshold: {LAB_MATCH_THRESHOLD}")

    while True:
        err = zed.grab()
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, resolution)
            frame_bgra = image_zed.get_data()
            current_frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            display_feed = current_frame_bgr.copy()
            status_text = ""
            if not start_frame_captured:
                 status_text = "Press 's' to capture start frame."
            elif not results_calculated:
                status_text = "Start captured. Move particles. Press 'e' for end."
            else:
                 status_text = "Displacement calculated (console). Press 'r' to reset."
            cv2.putText(display_feed, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if start_frame_captured:
                 cv2.putText(display_feed, f"Start: {len(start_particles)} found", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if results_calculated:
                 cv2.putText(display_feed, f"End: {len(end_particles)} found", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow(window_live, display_feed)
        else:
            print(f"Failed to grab frame: {err}. Exiting.")
            break
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            print("Quitting.")
            break
        elif key == ord("r"):
            print("Resetting capture.")
            start_image = None
            start_particles = []
            end_particles = []
            start_frame_captured = False
            start_vis_image = None
            results_calculated = False
            try: cv2.destroyWindow(window_start_detections)
            except cv2.error: pass
            print("Press 's' to capture start frame.")
        elif key == ord("s"):
            if not start_frame_captured or results_calculated:
                 if results_calculated:
                     print("Resetting for new capture...")
                     start_image = None
                     start_particles = []
                     end_particles = []
                     start_frame_captured = False
                     start_vis_image = None
                     results_calculated = False
                     try: cv2.destroyWindow(window_start_detections)
                     except cv2.error: pass
                 start_image = current_frame_bgr.copy()
                 print("Start frame captured. Detecting particles...")
                 start_particles = detect_particles_and_sample_color(start_image, blob_detector)
                 print(f" -> Found {len(start_particles)} blobs in start frame.")
                 start_frame_captured = True
                 results_calculated = False
                 if start_image is not None:
                     # Draw initial detections without displacements
                     start_vis_image = draw_detections(start_image, start_particles)
                     cv2.imshow(window_start_detections, start_vis_image)
                     print(f"Displaying start detections in '{window_start_detections}'.")
                 print("Move particles and press 'e' to capture end frame and calculate displacement.")
            else:
                 print("Start frame already captured. Move particles and press 'e' or press 'r' to reset.")
        elif key == ord("e"):
            if start_frame_captured and not results_calculated:
                end_image_capture = current_frame_bgr.copy()
                print("End frame captured. Detecting particles...")
                end_particles = detect_particles_and_sample_color(end_image_capture, blob_detector)
                print(f" -> Found {len(end_particles)} blobs in end frame.")
                print("\nMatching particles using Lab color distance...")
                matched_pairs, unmatched_start, unmatched_end = match_particles(start_particles, end_particles, LAB_MATCH_THRESHOLD)
                results_calculated = True

                # --- Create displacement map and print results --- #
                displacement_data = {}
                print("\n--- Displacement Results ---")
                if not matched_pairs:
                    print("No matching particles found.")
                else:
                    print(f"{len(matched_pairs)} particle(s) matched:")
                    for i, (start_p, end_p) in enumerate(matched_pairs):
                        start_pos, start_color_bgr, _ = start_p
                        end_pos, _, _ = end_p
                        sx, sy = start_pos
                        ex, ey = end_pos
                        dx = ex - sx
                        dy = ey - sy
                        displacement_data[start_pos] = (dx, dy) # Store displacement keyed by start pos
                        print(f"  Match {i+1}: Start@{start_pos} [Color(BGR):{start_color_bgr}] -> End@{end_pos} | Disp=({dx}, {dy})")
                if unmatched_start:
                    print(f"\n{len(unmatched_start)} unmatched start particle(s):")
                    for i, p in enumerate(unmatched_start):
                         print(f"  Unmatched Start {i+1}: Pos={p[0]}, Color(BGR)={p[1]}")
                if unmatched_end:
                    print(f"\n{len(unmatched_end)} unmatched end particle(s):")
                    for i, p in enumerate(unmatched_end):
                         print(f"  Unmatched End {i+1}: Pos={p[0]}, Color(BGR)={p[1]}")
                print("-----------------------------")
                print("Displaying vectors on start image. Press 'r' to start a new measurement.")

                # --- Save Start and End Images ---
                if start_image is not None:
                    cv2.imwrite("start_image.png", start_image)
                    print("Saved start_image.png")
                if end_image_capture is not None:
                    cv2.imwrite("end_image.png", end_image_capture)
                    print("Saved end_image.png")
                # ------------------------------------

                # --- Update and display start image with displacement vectors --- #
                if start_image is not None:
                    start_vis_image = draw_detections(start_image, start_particles, displacement_data)
                    cv2.imshow(window_start_detections, start_vis_image) # Refresh the window

            elif not start_frame_captured:
                print("Cannot calculate displacement. Capture start frame ('s') first.")
            else:
                print("Displacement already calculated. Press 'r' to reset for a new measurement.")
    cv2.destroyAllWindows()
    zed.close()
    print("ZED camera closed.")

if __name__ == "__main__":
    main()
