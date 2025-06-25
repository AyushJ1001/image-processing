import cv2
import numpy as np
import math
import argparse
import os

# Particle filter params
MIN_PARTICLE_AREA = 25
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
    """Auto-detect silver plate ROI"""
    global ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive thresholding to better detect the silver plate
    # Look for bright/light areas (silver plate) rather than dark areas
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter contours by area and aspect ratio to find the silver plate
        valid_contours = []
        image_area = image_bgr.shape[0] * image_bgr.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by area (should be a significant portion but not the whole image)
            if area > image_area * 0.05 and area < image_area * 0.8:
                # Filter by aspect ratio (roughly rectangular plate)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 3.0:  # Reasonable aspect ratio for a plate
                    # Filter by dimensions (should be reasonably sized)
                    if w > 100 and h > 100:
                        valid_contours.append((contour, area))
        
        if valid_contours:
            # Get the largest valid contour (most likely the silver plate)
            largest_contour = max(valid_contours, key=lambda x: x[1])[0]
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            ROI_X = x + ROI_MARGIN
            ROI_Y = y + ROI_MARGIN
            ROI_WIDTH = w - 2 * ROI_MARGIN
            ROI_HEIGHT = h - 2 * ROI_MARGIN
            
            return (ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT)
    
    return None

def detect_particles_and_sample_color(image_bgr, detector, depth_map=None, point_cloud=None, roi=None):
    """Detect blobs and get colors + positions (modified to work without ZED depth data)"""
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
            
            # Since we don't have ZED depth data, set world_pos to None
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
            # Convert to integers for OpenCV drawing functions
            end_point = (int(cx + dx), int(cy + dy))
            start_point = (int(cx), int(cy))
            cv2.arrowedLine(vis_image, start_point, end_point, (0, 0, 0), 3, tipLength=0.3)
            cv2.arrowedLine(vis_image, start_point, end_point, (255, 255, 255), 1, tipLength=0.3)

    return vis_image

def get_frames_from_video(video_path, frame_interval_seconds=1.0):
    """Extract two frames from video separated by specified interval"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * frame_interval_seconds)
    
    # Read first frame
    ret, start_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame from video")
    
    # Skip frames to get the second frame
    for _ in range(frame_interval):
        ret, end_frame = cap.read()
        if not ret:
            # If we can't skip enough frames, use the last available frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
            ret, end_frame = cap.read()
            if not ret:
                raise ValueError("Cannot read end frame from video")
            break
    
    cap.release()
    return start_frame, end_frame

def get_image_sequence(folder_path):
    """Get sorted list of image files from folder"""
    import glob
    
    # Common image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_files:
        raise ValueError(f"No image files found in folder: {folder_path}")
    
    # Sort files naturally (handles numeric sequences properly)
    image_files.sort()
    
    return image_files

def process_image_sequence(image_files, blob_detector):
    """Process sequence of images and accumulate displacement vectors"""
    if len(image_files) < 2:
        raise ValueError("Need at least 2 images for sequence processing")
    
    print(f"Processing sequence of {len(image_files)} images...")
    
    # Load first image to establish base
    base_image = cv2.imread(image_files[0])
    if base_image is None:
        raise ValueError(f"Cannot load image: {image_files[0]}")
    
    # Auto-detect ROI from first image
    roi = None
    if AUTO_DETECT_ROI:
        roi = detect_roi(base_image)
        if roi:
            print(f"ROI detected: {roi}")
    
    # Accumulate all displacement vectors
    all_displacements = {}  # Dictionary to store all displacement vectors
    sequence_info = []  # Store info about each pair processed
    
    for i in range(len(image_files) - 1):
        print(f"\nProcessing pair {i+1}/{len(image_files)-1}: {os.path.basename(image_files[i])} -> {os.path.basename(image_files[i+1])}")
        
        # Load consecutive images
        img1 = cv2.imread(image_files[i])
        img2 = cv2.imread(image_files[i+1])
        
        if img1 is None or img2 is None:
            print(f"Warning: Could not load image pair {i+1}, skipping...")
            continue
        
        # Detect particles in both images
        particles1 = detect_particles_and_sample_color(img1, blob_detector, roi=roi)
        particles2 = detect_particles_and_sample_color(img2, blob_detector, roi=roi)
        
        print(f"  Found {len(particles1)} -> {len(particles2)} particles")
        
        # Match particles
        matched_pairs, unmatched_start, unmatched_end = match_particles(
            particles1, particles2, 
            LAB_MATCH_THRESHOLD, 
            MAX_DISPLACEMENT_PIXELS, 
            None
        )
        
        print(f"  Matched {len(matched_pairs)} particles")
        
        # Add displacement vectors to accumulation
        pair_displacements = {}
        for start_p, end_p in matched_pairs:
            start_pos, start_color_bgr, _, _ = start_p
            end_pos, _, _, _ = end_p
            sx, sy = start_pos
            ex, ey = end_pos
            dx = ex - sx
            dy = ey - sy
            
            pair_displacements[start_pos] = (dx, dy)
            
            # Add to global accumulation (you could weight by color similarity, etc.)
            if start_pos in all_displacements:
                # Average with existing displacement
                old_dx, old_dy = all_displacements[start_pos]
                all_displacements[start_pos] = ((old_dx + dx) / 2, (old_dy + dy) / 2)
            else:
                all_displacements[start_pos] = (dx, dy)
        
        sequence_info.append({
            'pair': f"{os.path.basename(image_files[i])} -> {os.path.basename(image_files[i+1])}",
            'matches': len(matched_pairs),
            'displacements': pair_displacements
        })
    
    print(f"\n--- Sequence Processing Complete ---")
    print(f"Total unique displacement vectors accumulated: {len(all_displacements)}")
    
    # Get all particles from the base image for visualization
    base_particles = detect_particles_and_sample_color(base_image, blob_detector, roi=roi)
    
    # Create final visualization
    final_vis = draw_detections(base_image, base_particles, all_displacements, roi)
    
    return final_vis, base_image, all_displacements, sequence_info

def resize_image_for_display(image, max_width=1200, max_height=800):
    """Resize image for display if it's too large, maintaining aspect ratio"""
    height, width = image.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale, only downscale
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized, scale
    
    return image, 1.0

def process_images(start_image, end_image, blob_detector):
    """Process start and end images to detect particles and calculate displacement"""
    roi = None
    
    # Auto-detect ROI from start image if enabled
    if AUTO_DETECT_ROI:
        roi = detect_roi(start_image)
        if roi:
            print(f"ROI detected: {roi}")
    
    print("Detecting particles in start image...")
    start_particles = detect_particles_and_sample_color(start_image, blob_detector, roi=roi)
    print(f" -> Found {len(start_particles)} blobs in start frame.")
    
    print("Detecting particles in end image...")
    end_particles = detect_particles_and_sample_color(end_image, blob_detector, roi=roi)
    print(f" -> Found {len(end_particles)} blobs in end frame.")
    
    print("\nMatching particles using Lab color distance and spatial constraints...")
    matched_pairs, unmatched_start, unmatched_end = match_particles(
        start_particles, end_particles, 
        LAB_MATCH_THRESHOLD, 
        MAX_DISPLACEMENT_PIXELS, 
        None  # No 3D distance constraint since we don't have depth data
    )
    
    # Calculate and display results
    displacement_data = {}
    print("\n--- Displacement Results ---")
    if not matched_pairs:
        print("No matching particles found.")
    else:
        print(f"{len(matched_pairs)} particle(s) matched:")
        for i, (start_p, end_p) in enumerate(matched_pairs):
            start_pos, start_color_bgr, _, _ = start_p
            end_pos, _, _, _ = end_p
            sx, sy = start_pos
            ex, ey = end_pos
            dx = ex - sx
            dy = ey - sy
            
            displacement_data[start_pos] = (dx, dy)
            pixel_distance = math.sqrt(dx**2 + dy**2)
            print(f"  Match {i+1}: Start@{start_pos} [Color(BGR):{start_color_bgr}] -> End@{end_pos} | Disp=({dx}, {dy}) px | Distance: {pixel_distance:.1f} px")
    
    if unmatched_start:
        print(f"\n{len(unmatched_start)} unmatched start particle(s):")
        for i, p in enumerate(unmatched_start):
             print(f"  Unmatched Start {i+1}: Pos={p[0]}, Color(BGR)={p[1]}")
    if unmatched_end:
        print(f"\n{len(unmatched_end)} unmatched end particle(s):")
        for i, p in enumerate(unmatched_end):
             print(f"  Unmatched End {i+1}: Pos={p[0]}, Color(BGR)={p[1]}")
    print("-----------------------------")
    
    # Save images
    cv2.imwrite("start_image.png", start_image)
    cv2.imwrite("end_image.png", end_image)
    print("Saved start_image.png and end_image.png")
    
    # Create visualization
    start_vis_image = draw_detections(start_image, start_particles, displacement_data, roi)
    end_vis_image = draw_detections(end_image, end_particles, roi=roi)
    
    return start_vis_image, end_vis_image, matched_pairs

def main():
    parser = argparse.ArgumentParser(description='Image processing for particle displacement analysis')
    parser.add_argument('--mode', choices=['images', 'video', 'sequence'], required=True,
                        help='Processing mode: "images" for start/end image files, "video" for video file, "sequence" for image sequence in folder')
    parser.add_argument('--start', type=str, help='Path to start image (required for images mode)')
    parser.add_argument('--end', type=str, help='Path to end image (required for images mode)')
    parser.add_argument('--video', type=str, help='Path to video file (required for video mode)')
    parser.add_argument('--folder', type=str, help='Path to folder containing image sequence (required for sequence mode)')
    parser.add_argument('--interval', type=float, default=1.0, 
                        help='Time interval in seconds between frames for video mode (default: 1.0)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'images':
        if not args.start or not args.end:
            print("Error: --start and --end are required for images mode")
            return
        if not os.path.exists(args.start):
            print(f"Error: Start image file not found: {args.start}")
            return
        if not os.path.exists(args.end):
            print(f"Error: End image file not found: {args.end}")
            return
    elif args.mode == 'video':
        if not args.video:
            print("Error: --video is required for video mode")
            return
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            return
    elif args.mode == 'sequence':
        if not args.folder:
            print("Error: --folder is required for sequence mode")
            return
        if not os.path.exists(args.folder):
            print(f"Error: Folder not found: {args.folder}")
            return
        if not os.path.isdir(args.folder):
            print(f"Error: Path is not a directory: {args.folder}")
            return
    
    blob_detector = setup_blob_detector()
    
    try:
        if args.mode == 'images':
            print(f"Loading images: {args.start} and {args.end}")
            start_image = cv2.imread(args.start)
            end_image = cv2.imread(args.end)
            
            if start_image is None:
                print(f"Error: Cannot load start image: {args.start}")
                return
            if end_image is None:
                print(f"Error: Cannot load end image: {args.end}")
                return
            
            # Process the images
            start_vis, end_vis, matches = process_images(start_image, end_image, blob_detector)
            
            # Resize images for display if they're too large
            start_display, start_scale = resize_image_for_display(start_vis)
            end_display, end_scale = resize_image_for_display(end_vis)
            
            # Display results
            print(f"\nDisplaying results. Found {len(matches)} particle matches.")
            if start_scale < 1.0 or end_scale < 1.0:
                print(f"Images resized for display: Start={start_scale:.2f}x, End={end_scale:.2f}x")
            print("Press any key to close windows and exit.")
            
            # Create resizable windows
            cv2.namedWindow("Start Frame with Displacement Vectors", cv2.WINDOW_NORMAL)
            cv2.namedWindow("End Frame Detections", cv2.WINDOW_NORMAL)
            
            cv2.imshow("Start Frame with Displacement Vectors", start_display)
            cv2.imshow("End Frame Detections", end_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                
        elif args.mode == 'video':
            print(f"Extracting frames from video: {args.video} (interval: {args.interval}s)")
            start_image, end_image = get_frames_from_video(args.video, args.interval)
            
            # Process the images
            start_vis, end_vis, matches = process_images(start_image, end_image, blob_detector)
            
            # Resize images for display if they're too large
            start_display, start_scale = resize_image_for_display(start_vis)
            end_display, end_scale = resize_image_for_display(end_vis)
            
            # Display results
            print(f"\nDisplaying results. Found {len(matches)} particle matches.")
            if start_scale < 1.0 or end_scale < 1.0:
                print(f"Images resized for display: Start={start_scale:.2f}x, End={end_scale:.2f}x")
            print("Press any key to close windows and exit.")
            
            # Create resizable windows
            cv2.namedWindow("Start Frame with Displacement Vectors", cv2.WINDOW_NORMAL)
            cv2.namedWindow("End Frame Detections", cv2.WINDOW_NORMAL)
            
            cv2.imshow("Start Frame with Displacement Vectors", start_display)
            cv2.imshow("End Frame Detections", end_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        elif args.mode == 'sequence':
            print(f"Processing image sequence from folder: {args.folder}")
            image_files = get_image_sequence(args.folder)
            print(f"Found {len(image_files)} images in sequence")
            
            # Process the sequence
            final_vis, base_image, all_displacements, sequence_info = process_image_sequence(image_files, blob_detector)
            
            # Print sequence summary
            print(f"\n--- Sequence Summary ---")
            for info in sequence_info:
                print(f"  {info['pair']}: {info['matches']} matches")
            
            # Save result
            output_filename = f"sequence_displacement_field.png"
            cv2.imwrite(output_filename, final_vis)
            print(f"\nSaved cumulative displacement field to: {output_filename}")
            
            # Resize for display
            display_image, scale = resize_image_for_display(final_vis)
            
            # Display results
            print(f"\nDisplaying cumulative displacement field with {len(all_displacements)} vectors.")
            if scale < 1.0:
                print(f"Image resized for display: {scale:.2f}x")
            print("Press any key to close window and exit.")
            
            # Create resizable window
            cv2.namedWindow("Cumulative Displacement Field", cv2.WINDOW_NORMAL)
            cv2.imshow("Cumulative Displacement Field", display_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
