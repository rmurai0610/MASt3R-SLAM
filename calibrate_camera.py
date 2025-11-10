import cv2
import numpy as np
import yaml
import os
import time

class AutoChessboardCalibrator:
    def __init__(self):
        # ----------------------------------------------------
        # Chessboard settings for A4 board:
        #  - 11x8 squares → 10x7 inner corners (vertices)
        #  - Each square is 25 mm
        # ----------------------------------------------------
        self.chessboard_size = (10, 7)  # inner corners: 10 per row, 7 per column
        self.square_size = 25.0         # size of each square in millimeters

        # Prepare object points: (0,0,0), (1,0,0), ... scaled by square size
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0],
                                     0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size

        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane

        self.capture_count = 0
        self.min_captures = 5  # Minimum images required for calibration

        # To avoid duplicate captures, add a delay (in seconds)
        self.last_capture_time = 0
        self.capture_delay = 1.0  # seconds

        # Calibration results (to be set after calibration)
        self.camera_matrix = None
        self.dist_coeffs = None

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open video capture")
            return

        window_name = "Auto Chessboard Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("Starting automatic calibration. Show the chessboard to the camera.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Try to find the chessboard corners in the current frame
            found, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            if found:
                # Refine the corner locations for higher accuracy
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(frame, self.chessboard_size, corners, found)

                current_time = time.time()
                # Only capture if enough time has passed since the last capture
                if current_time - self.last_capture_time > self.capture_delay:
                    self.objpoints.append(self.objp)
                    self.imgpoints.append(corners)
                    self.capture_count += 1
                    self.last_capture_time = current_time
                    print(f"Auto-captured frame #{self.capture_count}")

            # Display capture status on the video feed
            status_text = f"Captured: {self.capture_count}/{self.min_captures}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press ESC to quit manually", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC key to exit manually
                print("Manual exit triggered")
                break

            # If we have enough captures, break the loop and calibrate
            if self.capture_count >= self.min_captures:
                print("Required captures obtained. Calibrating...")
                break

        cap.release()
        cv2.destroyAllWindows()

        if self.capture_count < self.min_captures:
            print("Not enough valid captures for calibration. Exiting.")
            return

        # ----------------------------------------------------
        # Perform camera calibration using the captured images
        # ----------------------------------------------------
        ret_calib, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, gray.shape[::-1], None, None
        )
        if not ret_calib:
            print("Calibration failed.")
            return

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        print("Calibration succeeded.")
        print("Camera Matrix:")
        print(camera_matrix)
        print("Distortion Coefficients:")
        print(dist_coeffs)

        # Save calibration data in the requested YAML format
        self.save_calibration(frame)
        print("Calibration process completed.")

    def save_calibration(self, frame):
        # Convert intrinsic parameters to native Python types
        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])
        # Flatten the distortion coefficients to native floats.
        dist_list = [float(x) for x in self.dist_coeffs.flatten().tolist()]
        # Combine into one list.
        calibration_list = [fx, fy, cx, cy] + dist_list

        # Get image dimensions
        height, width = frame.shape[:2]

        # Prepare the output string manually to match the desired format.
        # Note: This will output the list inline exactly as you specified.
        output = (
            f"width: {width}\n"
            f"height: {height}\n"
            "# With distortion (fx, fy, cx, cy, k1, k2, p1, p2)\n"
            f"calibration:  {calibration_list}\n"
        )

        # Save to "config/intrinsics.yaml" in the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "config")
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, "intrinsics.yaml")

        with open(output_filename, "w") as f:
            f.write(output)

        print(f"Calibration data saved to {output_filename}")

if __name__ == '__main__':
    calibrator = AutoChessboardCalibrator()
    calibrator.run()
