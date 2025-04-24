import cv2
import numpy as np
import pyzed.sl as sl
import math

# Particle filter params
MIN_PARTICLE_AREA = 15
MAX_PARTICLE_AREA = 200
MIN_CIRCULARITY = 0.6
MAX_CIRCULARITY = 0.95

# Blob detector params
BLOB_MIN_THRESHOLD = 10
BLOB_MAX_THRESHOLD = 250
BLOB_MIN_CONVEXITY = 0.80
BLOB_MIN_INERTIA_RATIO = 0.1

# Color matching threshold
LAB_MATCH_THRESHOLD = 50.0

# Max displacement limits
MAX_DISPLACEMENT_PIXELS = 50
MAX_DISPLACEMENT_MM = 50.0

# ROI params
ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT = 0, 0, 0, 0
AUTO_DETECT_ROI = True
ROI_MARGIN = 20

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

def detect_roi(image_bgr):
    """Auto-detect black board ROI"""
    global ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        ROI_X = x + ROI_MARGIN
        ROI_Y = y + ROI_MARGIN
        ROI_WIDTH = w - 2 * ROI_MARGIN
        ROI_HEIGHT = h - 2 * ROI_MARGIN
        
        return (ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT)
    return None

def detect_particles_and_sample_color(image_bgr, detector, depth_map=None, point_cloud=None, roi=None):
    """Detect blobs and get colors + positions"""
    detected_list = []
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    gray_opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    roi_mask = None
    if roi:
        roi_x, roi_y, roi_w, roi_h = roi
        roi_mask = np.zeros_like(gray_opened)
        roi_mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = 255
        gray_opened = cv2.bitwise_and(gray_opened, gray_opened, mask=roi_mask)

    keypoints = detector.detect(gray_opened)

    for kp in keypoints:
        cX = int(kp.pt[0])
        cY = int(kp.pt[1])
        
        if roi and (cX < roi[0] or cY < roi[1] or cX >= roi[0]+roi[2] or cY >= roi[1]+roi[3]):
            continue
            
        if 0 <= cX < width and 0 <= cY < height:
            color_bgr = tuple(map(int, image_bgr[cY, cX]))

            pixel_bgr = np.uint8([[color_bgr]])
            pixel_lab = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2LAB)
            color_lab = tuple(map(int, pixel_lab[0, 0]))
            
            world_pos = None
            if depth_map is not None and point_cloud is not None:
                if 0 <= cY < depth_map.get_height() and 0 <= cX < depth_map.get_width():
                    err, world_pos = point_cloud.get_value(cX, cY)
                    if err != sl.ERROR_CODE.SUCCESS:
                        world_pos = None

            detected_list.append(((cX, cY), color_bgr, color_lab, world_pos))
    return detected_list

def lab_color_distance(color1_lab, color2_lab):
    """Get Lab color distance"""
    l1, a1, b1 = color1_lab
    l2, a2, b2 = color2_lab
    return math.sqrt((l1 - l2)**2 + (a1 - a2)**2 + (b1 - b2)**2)

def spatial_distance(pos1, pos2):
    """Get 2D distance"""
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def spatial_distance_3d(pos1, pos2):
    """Get 3D distance"""
    if pos1 is None or pos2 is None:
        return float('inf')
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)

def match_particles(start_particles, end_particles, max_lab_dist, max_pixel_dist=None, max_3d_dist=None):
    """Match particles between frames"""
    num_start = len(start_particles)
    num_end = len(end_particles)
    if num_start == 0 or num_end == 0:
        return [], start_particles, end_particles

    potential_matches = []
    for i in range(num_start):
        start_pos = start_particles[i][0]
        start_lab = start_particles[i][2]
        start_world = start_particles[i][3]
        
        for j in range(num_end):
            end_pos = end_particles[j][0]
            end_lab = end_particles[j][2]
            end_world = end_particles[j][3]
            
            color_dist = lab_color_distance(start_lab, end_lab)
            pixel_dist = spatial_distance(start_pos, end_pos)
            world_dist = spatial_distance_3d(start_world, end_world) * 1000
            
            if color_dist <= max_lab_dist:
                if (max_pixel_dist is None or pixel_dist <= max_pixel_dist) and \
                   (max_3d_dist is None or world_dist <= max_3d_dist or math.isnan(world_dist)):
                    color_score = color_dist / max_lab_dist
                    space_score = 0
                    if max_pixel_dist:
                        space_score = pixel_dist / max_pixel_dist
                    elif max_3d_dist and not math.isnan(world_dist):
                        space_score = world_dist / max_3d_dist
                    
                    combined_score = 0.3 * color_score + 0.7 * space_score
                    potential_matches.append((combined_score, i, j))

    potential_matches.sort()
    matched_pairs = []
    matched_start_indices = set()
    matched_end_indices = set()
    for _, start_idx, end_idx in potential_matches:
        if start_idx not in matched_start_indices and end_idx not in matched_end_indices:
            matched_pairs.append((start_particles[start_idx], end_particles[end_idx]))
            matched_start_indices.add(start_idx)
            matched_end_indices.add(end_idx)

    unmatched_start = [p for i, p in enumerate(start_particles) if i not in matched_start_indices]
    unmatched_end = [p for i, p in enumerate(end_particles) if i not in matched_end_indices]

    return matched_pairs, unmatched_start, unmatched_end

def draw_detections(image, particles_list, displacements_map=None, roi=None):
    """Draw detections and vectors"""
    vis_image = image.copy()
    
    if roi:
        roi_x, roi_y, roi_w, roi_h = roi
        cv2.rectangle(vis_image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    
    for particle_data in particles_list:
        centroid, color_bgr, _, _ = particle_data
        cx, cy = centroid
        draw_color = tuple(map(int, color_bgr))
        cv2.circle(vis_image, (cx, cy), 7, draw_color, -1)
        cv2.circle(vis_image, (cx, cy), 7, (0, 0, 0), 1)

        if displacements_map and centroid in displacements_map:
            dx, dy = displacements_map[centroid]
            end_point = (cx + dx, cy + dy)
            cv2.arrowedLine(vis_image, (cx, cy), end_point, (0, 0, 0), 3, tipLength=0.3)
            cv2.arrowedLine(vis_image, (cx, cy), end_point, (255, 255, 255), 1, tipLength=0.3)

    return vis_image

def main():
    blob_detector = setup_blob_detector()
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open failed: {status}")
        zed.close()
        exit(1)
    cam_info = zed.get_camera_information()
    resolution = cam_info.camera_configuration.resolution
    
    image_zed = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.U8_C4)
    depth_zed = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C1)
    point_cloud = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.F32_C4)
    
    start_image = None
    start_particles = []
    end_particles = []
    start_frame_captured = False
    start_vis_image = None
    results_calculated = False
    roi = None
    window_live = "ZED Live Feed"
    window_start_detections = "Start Frame Detections"
    print("Camera Initialized. Press 's' -> move particles -> 'e'. 'r' to reset, 'q' to quit.")
    print(f"Using Lab Color Matching Threshold: {LAB_MATCH_THRESHOLD}")
    print(f"Maximum Displacement Constraints: {MAX_DISPLACEMENT_PIXELS} pixels, {MAX_DISPLACEMENT_MM} mm")

    while True:
        err = zed.grab()
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, resolution)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH, sl.MEM.CPU, resolution)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, resolution)
            
            frame_bgra = image_zed.get_data()
            current_frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            display_feed = current_frame_bgr.copy()
            
            if AUTO_DETECT_ROI and roi is None:
                roi = detect_roi(current_frame_bgr)
                if roi:
                    print(f"ROI detected: {roi}")
            
            if roi:
                roi_x, roi_y, roi_w, roi_h = roi
                cv2.rectangle(display_feed, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            
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
                 
                 start_particles = detect_particles_and_sample_color(
                     start_image, blob_detector, depth_zed, point_cloud, roi
                 )
                 print(f" -> Found {len(start_particles)} blobs in start frame.")
                 start_frame_captured = True
                 results_calculated = False
                 if start_image is not None:
                     start_vis_image = draw_detections(start_image, start_particles, roi=roi)
                     cv2.imshow(window_start_detections, start_vis_image)
                     print(f"Displaying start detections in '{window_start_detections}'.")
                 print("Move particles and press 'e' to capture end frame and calculate displacement.")
            else:
                 print("Start frame already captured. Move particles and press 'e' or press 'r' to reset.")
        elif key == ord("e"):
            if start_frame_captured and not results_calculated:
                end_image_capture = current_frame_bgr.copy()
                print("End frame captured. Detecting particles...")
                
                end_particles = detect_particles_and_sample_color(
                    end_image_capture, blob_detector, depth_zed, point_cloud, roi
                )
                print(f" -> Found {len(end_particles)} blobs in end frame.")
                print("\nMatching particles using Lab color distance and spatial constraints...")
                matched_pairs, unmatched_start, unmatched_end = match_particles(
                    start_particles, end_particles, 
                    LAB_MATCH_THRESHOLD, 
                    MAX_DISPLACEMENT_PIXELS, 
                    MAX_DISPLACEMENT_MM
                )
                results_calculated = True

                displacement_data = {}
                print("\n--- Displacement Results ---")
                if not matched_pairs:
                    print("No matching particles found.")
                else:
                    print(f"{len(matched_pairs)} particle(s) matched:")
                    for i, (start_p, end_p) in enumerate(matched_pairs):
                        start_pos, start_color_bgr, _, start_world = start_p
                        end_pos, _, _, end_world = end_p
                        sx, sy = start_pos
                        ex, ey = end_pos
                        dx = ex - sx
                        dy = ey - sy
                        
                        world_disp_str = "N/A"
                        if start_world is not None and end_world is not None:
                            dx_mm = (end_world[0] - start_world[0]) * 1000
                            dy_mm = (end_world[1] - start_world[1]) * 1000
                            dz_mm = (end_world[2] - start_world[2]) * 1000
                            dist_mm = math.sqrt(dx_mm**2 + dy_mm**2 + dz_mm**2)
                            world_disp_str = f"3D: ({dx_mm:.1f}, {dy_mm:.1f}, {dz_mm:.1f}) mm, Dist: {dist_mm:.1f} mm"
                        
                        displacement_data[start_pos] = (dx, dy)
                        print(f"  Match {i+1}: Start@{start_pos} [Color(BGR):{start_color_bgr}] -> End@{end_pos} | Disp=({dx}, {dy}) px | {world_disp_str}")
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

                if start_image is not None:
                    cv2.imwrite("start_image.png", start_image)
                    print("Saved start_image.png")
                if end_image_capture is not None:
                    cv2.imwrite("end_image.png", end_image_capture)
                    print("Saved end_image.png")

                if start_image is not None:
                    start_vis_image = draw_detections(start_image, start_particles, displacement_data, roi)
                    cv2.imshow(window_start_detections, start_vis_image)

            elif not start_frame_captured:
                print("Cannot calculate displacement. Capture start frame ('s') first.")
            else:
                print("Displacement already calculated. Press 'r' to reset for a new measurement.")
    cv2.destroyAllWindows()
    zed.close()
    print("ZED camera closed.")

if __name__ == "__main__":
    main()
