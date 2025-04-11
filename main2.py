import cv2
import numpy as np
import pyzed.sl as sl # Re-added pyzed import

# --- Color Definitions (Adjust these HSV ranges for your specific colors/lighting) ---
# Format: "ColorName": ([Lower_H, Lower_S, Lower_V], [Upper_H, Upper_S, Upper_V])
# You can use tools like https://alloyui.com/examples/color-picker/hsv.html to find ranges
COLOR_RANGES_HSV = {
    "red": ([0, 120, 70], [10, 255, 255]), # Example range for red
    "green": ([36, 100, 100], [86, 255, 255]), # Example range for green
    "blue": ([94, 100, 100], [124, 255, 255]), # Example range for blue
    # Add more colors as needed
}

# BGR colors for drawing (match keys in COLOR_RANGES_HSV)
DRAW_COLORS_BGR = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
}

# --- Parameters for Particle Filtering ---
MIN_PARTICLE_AREA = 20 # Adjust based on expected particle size in pixels
MAX_PARTICLE_AREA = 500 # Adjust based on expected particle size in pixels

def detect_colored_particles(image, color_ranges_hsv):
    """
    Detects particles based on color ranges in an image.

    Args:
        image (np.ndarray): Input BGR image.
        color_ranges_hsv (dict): Dictionary of color names to HSV range tuples.
                                 e.g., {"red": ([H_low, S_low, V_low], [H_high, S_high, V_high])}

    Returns:
        dict: A dictionary mapping color names to detected particle centroids.
              e.g., {"red": (x1, y1), "blue": (x2, y2)}
              Returns None for a color if no particle is found.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    detected_particles = {}

    for color_name, (lower_hsv, upper_hsv) in color_ranges_hsv.items():
        lower_bound = np.array(lower_hsv)
        upper_bound = np.array(upper_hsv)

        # Create a mask for the current color
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Optional: Apply morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found_particle = False
        best_contour = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter contours by area
            if MIN_PARTICLE_AREA < area < MAX_PARTICLE_AREA:
                # If multiple contours fit, maybe take the largest within range?
                # Or handle multiple particles of the same color differently.
                # For now, take the first valid one. Refined logic might be needed.
                # Let's take the largest valid one for now.
                if area > max_area:
                    max_area = area
                    best_contour = contour
                    found_particle = True

        if found_particle and best_contour is not None:
            # Calculate centroid using moments
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                detected_particles[color_name] = (cX, cY)
            else:
                detected_particles[color_name] = None # Mark as not found if moment is zero
        else:
            detected_particles[color_name] = None # Indicate not found

    return detected_particles

def calculate_displacements(start_particles, end_particles):
    """
    Calculates displacement vectors for particles matched by color.

    Args:
        start_particles (dict): Detected particles from the start image.
                                {"color_name": (x, y) or None}
        end_particles (dict): Detected particles from the end image.
                              {"color_name": (x, y) or None}

    Returns:
        dict: A dictionary mapping color names to displacement vectors.
              e.g., {"red": (dx, dy), "blue": (dx, dy)}
              Returns None for a color if not found in both images.
    """
    displacements = {}
    for color_name, start_pos in start_particles.items():
        end_pos = end_particles.get(color_name, None)

        if start_pos is not None and end_pos is not None:
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            displacements[color_name] = (dx, dy)
        else:
            displacements[color_name] = None # Cannot calculate displacement

    return displacements

def draw_results(image, particles, displacements, color_map_bgr):
    """Draws detected centroids and displacement vectors."""
    vis_image = image.copy()
    for color_name, position in particles.items():
        if position is not None:
            bgr_color = color_map_bgr.get(color_name, (0, 255, 0)) # Default green
            cx, cy = position
            # Draw centroid
            cv2.circle(vis_image, (cx, cy), 5, bgr_color, -1)
            cv2.putText(vis_image, color_name, (cx - 15, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 1)

            # Draw displacement vector if available and requested (displacements not empty)
            if displacements:
                displacement = displacements.get(color_name)
                if displacement is not None:
                    dx, dy = displacement
                    end_point = (cx + dx, cy + dy) # End point relative to start image's centroid
                    # Draw arrow on the start image visualization
                    cv2.arrowedLine(vis_image, (cx, cy), (cx + dx, cy + dy), (255, 255, 255), 2, tipLength=0.3)
                    cv2.putText(vis_image, f"({dx},{dy})", (cx + 5, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return vis_image

def main():
    # --- ZED Camera Initialization ---
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    # Use a resolution appropriate for seeing your particles clearly
    # Lower resolutions might increase frame rate if needed
    init_params.camera_resolution = sl.RESOLUTION.HD720 # Example, adjust as needed
    init_params.camera_fps = 30 # Example, adjust as needed
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open failed: {status}")
        zed.close()
        exit(1)

    # Retrieve camera resolution and create image object
    cam_info = zed.get_camera_information()
    resolution = cam_info.camera_configuration.resolution
    image_zed = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.U8_C4) # BGRA

    # --- State Variables ---
    start_image = None
    end_image = None
    start_particles = None
    end_particles = None
    displacements = None
    start_frame_captured = False
    results_ready = False
    result_display_image = None

    window_live = "ZED Live Feed"
    window_results = "Displacement Results"

    print("Camera Initialized. Press 's' to capture start frame, 'e' for end frame, 'r' to reset, 'q' to quit.")

    while True:
        err = zed.grab()
        if err == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, resolution)
            # Get numpy array (BGRA) and convert to BGR for OpenCV processing
            frame_bgra = image_zed.get_data()
            current_frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            # Display the live feed
            display_feed = current_frame_bgr.copy()
            status_text = ""
            if start_frame_captured and not results_ready:
                status_text = "Start frame captured. Press 'e' for end frame."
            elif results_ready:
                status_text = "Results ready. Press 'r' to reset."
            else:
                status_text = "Press 's' to capture start frame."

            cv2.putText(display_feed, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window_live, display_feed)

        else:
            print(f"Failed to grab frame: {err}. Exiting.")
            break # Exit if frame grab fails

        key = cv2.waitKey(10) & 0xFF # Increased wait key delay slightly

        if key == ord("q"):
            print("Quitting.")
            break
        elif key == ord("r"):
            print("Resetting capture.")
            start_image = None
            end_image = None
            start_particles = None
            end_particles = None
            displacements = None
            start_frame_captured = False
            results_ready = False
            result_display_image = None
            try:
                cv2.destroyWindow(window_results) # Close result window if open
            except cv2.error:
                pass # Ignore error if window doesn't exist
            print("Press 's' to capture start frame.")
        elif key == ord("s"):
            if not start_frame_captured:
                start_image = current_frame_bgr.copy()
                print("Start frame captured. Detecting particles...")
                start_particles = detect_colored_particles(start_image, COLOR_RANGES_HSV)
                print(f" -> Start particles found: {start_particles}")
                start_frame_captured = True
                results_ready = False # Ensure results are not marked ready yet
                print("Move particles and press 'e' to capture end frame.")
            else:
                print("Start frame already captured. Press 'r' to reset first.")
        elif key == ord("e"):
            if start_frame_captured and not results_ready:
                end_image = current_frame_bgr.copy()
                print("End frame captured. Detecting particles...")
                end_particles = detect_colored_particles(end_image, COLOR_RANGES_HSV)
                print(f" -> End particles found: {end_particles}")

                print("Calculating displacements...")
                displacements = calculate_displacements(start_particles, end_particles)
                print(f" -> Displacements: {displacements}")

                # Prepare visualization on the start image
                print("Preparing result visualization...")
                if start_image is not None:
                    result_display_image = draw_results(start_image, start_particles, displacements, DRAW_COLORS_BGR)
                    results_ready = True
                    print(f"Results ready. Displaying in '{window_results}'. Press 'r' to reset.")
                else: # Should not happen if start_frame_captured is true, but safety check
                    print("Error: Start image was lost somehow.")
                    results_ready = False

            elif not start_frame_captured:
                print("Cannot capture end frame. Capture start frame ('s') first.")
            # else: results_ready is True, do nothing until reset

        # Display results window if ready
        if results_ready and result_display_image is not None:
            cv2.imshow(window_results, result_display_image)

    # --- Cleanup ---
    cv2.destroyAllWindows()
    zed.close()
    print("ZED camera closed.")

if __name__ == "__main__":
    main()
