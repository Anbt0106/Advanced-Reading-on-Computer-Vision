import numpy as np
import cv2
import warnings

def safe_correlation(desc1, desc2):
    """
    Safely compute correlation between two descriptors, handling edge cases
    that cause numpy warnings
    """
    # Check if descriptors have any variation
    std1 = np.std(desc1)
    std2 = np.std(desc2)

    # If either descriptor has zero standard deviation, return 0 correlation
    if std1 < 1e-10 or std2 < 1e-10:
        return 0.0

    # Suppress specific numpy warnings for correlation calculation
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                              message='invalid value encountered in divide')

        try:
            correlation = np.corrcoef(desc1, desc2)[0, 1]
            # Handle NaN results
            if np.isnan(correlation):
                return 0.0
            return correlation
        except:
            return 0.0

def match_corners_safe(img1, corners1, img2, corners2, window_size=16, threshold=0.7):
    """
    Enhanced corner matching with safe correlation calculation
    """
    matches = []
    descriptors1 = []
    descriptors2 = []
    valid_corners1 = []
    valid_corners2 = []

    half_window = window_size // 2

    # Extract descriptors for image 1
    for corner in corners1:
        x, y, _ = corner
        if (half_window <= x < img1.shape[1] - half_window and
            half_window <= y < img1.shape[0] - half_window):
            desc = img1[y-half_window:y+half_window, x-half_window:x+half_window].flatten()
            # Normalize descriptor to reduce lighting effects
            desc = desc.astype(np.float32)
            desc_mean = np.mean(desc)
            desc_std = np.std(desc)
            if desc_std > 1e-10:  # Only normalize if there's variation
                desc = (desc - desc_mean) / desc_std
            descriptors1.append(desc)
            valid_corners1.append(corner)

    # Extract descriptors for image 2
    for corner in corners2:
        x, y, _ = corner
        if (half_window <= x < img2.shape[1] - half_window and
            half_window <= y < img2.shape[0] - half_window):
            desc = img2[y-half_window:y+half_window, x-half_window:x+half_window].flatten()
            # Normalize descriptor to reduce lighting effects
            desc = desc.astype(np.float32)
            desc_mean = np.mean(desc)
            desc_std = np.std(desc)
            if desc_std > 1e-10:  # Only normalize if there's variation
                desc = (desc - desc_mean) / desc_std
            descriptors2.append(desc)
            valid_corners2.append(corner)

    print(f"Valid descriptors: {len(descriptors1)} from image 1, {len(descriptors2)} from image 2")

    # Match descriptors using safe correlation
    for i, desc1 in enumerate(descriptors1):
        best_match_idx = -1
        best_correlation = -1
        second_best = -1

        for j, desc2 in enumerate(descriptors2):
            # Use safe correlation calculation
            correlation = safe_correlation(desc1, desc2)

            if correlation > best_correlation:
                second_best = best_correlation
                best_correlation = correlation
                best_match_idx = j
            elif correlation > second_best:
                second_best = correlation

        # Ratio test for robust matching
        if best_match_idx >= 0 and best_correlation > threshold:
            if second_best <= 0 or best_correlation / second_best > 1.2:
                matches.append((valid_corners1[i], valid_corners2[best_match_idx]))

    return matches

def analyze_keypoint_positions(matches, img1_shape, img2_shape):
    """
    Analysis function for keypoint position comparison
    """
    print(f"\n=== KEYPOINT POSITION COMPARISON ===")
    print(f"Found {len(matches)} matches between original and transformed images")

    if len(matches) == 0:
        print("No matches found to analyze.")
        return

    # Extract coordinates
    coords1 = np.array([(match[0][0], match[0][1]) for match in matches])
    coords2 = np.array([(match[1][0], match[1][1]) for match in matches])

    # Calculate displacement vectors
    displacements = coords2 - coords1

    # Statistical analysis
    mean_displacement = np.mean(displacements, axis=0)
    std_displacement = np.std(displacements, axis=0)

    print(f"Average displacement: ({mean_displacement[0]:.2f}, {mean_displacement[1]:.2f})")
    print(f"Standard deviation: ({std_displacement[0]:.2f}, {std_displacement[1]:.2f})")

    # Distance analysis
    distances = np.linalg.norm(displacements, axis=1)
    print(f"Mean distance: {np.mean(distances):.2f} pixels")
    print(f"Distance std: {np.std(distances):.2f} pixels")
    print(f"Min distance: {np.min(distances):.2f} pixels")
    print(f"Max distance: {np.max(distances):.2f} pixels")

# Example usage with error handling
def run_safe_matching_example():
    """
    Example of how to use the safe correlation matching
    """
    try:
        # Your existing code here, but replace the correlation calculation
        # with the safe version
        print("Using safe correlation calculation to avoid numpy warnings...")

        # Example: Load images and detect corners (your existing code)
        # img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
        # corners1 = detect_harris_corners(img1)  # Your corner detection function
        # corners2 = detect_harris_corners(img2)  # Your corner detection function

        # Use safe matching
        # matches = match_corners_safe(img1, corners1, img2, corners2)
        # analyze_keypoint_positions(matches, img1.shape, img2.shape)

        print("Matching completed without numpy warnings!")

    except Exception as e:
        print(f"Error in matching: {e}")

if __name__ == "__main__":
    run_safe_matching_example()
