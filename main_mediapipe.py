import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import UnivariateSpline

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# Input and output video paths
input_video_path = "sources/videos/dead_bug.MOV"
output_video_path = "outputs/videos/main_mediapipe.MOV"

# Open the video file
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# List to store curvature data
curvature_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image.flags.writeable = True

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract shoulder and hip landmarks (normalized coordinates)
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_width,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_width,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * frame_width,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * frame_height]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * frame_width,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * frame_height]

        # Calculate midpoints for shoulders and hips
        shoulder_mid = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
        hip_mid = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]

        # Approximate spine midpoint (optional, between shoulders and hips)
        spine_mid = [(shoulder_mid[0] + hip_mid[0]) / 2, (shoulder_mid[1] + hip_mid[1]) / 2]

        # Points for curvature estimation
        points = np.array([shoulder_mid, spine_mid, hip_mid])
        x_coords, y_coords = points[:, 0], points[:, 1]

        # Fit a quadratic curve to estimate curvature
        try:
            # Sort points by y-coordinate to ensure proper spline fitting
            sorted_indices = np.argsort(y_coords)
            x_sorted = x_coords[sorted_indices]
            y_sorted = y_coords[sorted_indices]
            spline = UnivariateSpline(y_sorted, x_sorted, k=2, s=0)  # Quadratic spline
            curvature = spline.derivative(n=2)(y_sorted).mean()  # Average second derivative
            curvature_data.append(curvature)

            # Visualize the spline (optional)
            y_range = np.linspace(min(y_sorted), max(y_sorted), 100)
            x_spline = spline(y_range)
            for i in range(len(y_range) - 1):
                pt1 = (int(x_spline[i]), int(y_range[i]))
                pt2 = (int(x_spline[i + 1]), int(y_range[i + 1]))
                cv2.line(image, pt1, pt2, (255, 0, 0), 2)  # Blue line for curve
        except:
            curvature_data.append(0)  # Fallback if spline fails

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        # Display curvature value on frame
        curvature_text = f"Curvature: {curvature:.4f}"
        cv2.putText(image, curvature_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Write frame to output video
    out.write(image)

    # Optional: Display frame
    cv2.imshow('Cat-Cow Pose Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()
print(f"Processed video saved as {output_video_path}")

# Save curvature data to a file (optional)
np.savetxt('curvature_data.txt', curvature_data)
print("Curvature data saved to curvature_data.txt")