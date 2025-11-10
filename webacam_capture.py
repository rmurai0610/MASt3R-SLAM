import cv2
import datetime
import os

def main():
    # Generate a timestamp for the filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = "/home/rudra/Desktop/MASt3R-SLAM/webcam"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    output_path = os.path.join(output_dir, f"{timestamp}.mp4")
    
    # Open the default webcam (device 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    # Get frame dimensions and attempt to read the FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30  # fallback to 30 FPS if not available

    # Define the codec and create the VideoWriter object
    # 'mp4v' is generally used for mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Recording video. Press 'q' to stop recording.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Write the current frame to the video file
        out.write(frame)

        # Optionally, display the frame in a window
        cv2.imshow("Recording", frame)
        
        # Check if the user pressed 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping recording.")
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to: {output_path}")

if __name__ == '__main__':
    main()
