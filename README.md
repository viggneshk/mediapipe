# MediaPipe Pose Landmarker Web Application

This is a Python-based web application that processes videos using MediaPipe's Pose Landmarker to detect and visualize pose landmarks.

## Features

- Upload videos through a web interface
- Process videos using MediaPipe's Pose Landmarker (heavy model)
- Visualize 33 pose landmarks on the video
- Download the processed video with landmark visualization

## Requirements

- Python 3.7+
- MediaPipe
- Flask
- OpenCV

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```
   pip install mediapipe flask opencv-python
   ```

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Upload a video file (supported formats: MP4, AVI, MOV, WEBM)

4. Wait for the processing to complete (this may take some time depending on the video length)

5. View and download the processed video with pose landmarks visualized

## How It Works

1. The application uses Flask to provide a web interface for uploading videos
2. When a video is uploaded, it's saved to the server temporarily
3. MediaPipe's Pose Landmarker (heavy model) processes the video frame by frame
4. The pose landmarks are visualized on each frame
5. The processed frames are compiled into a new video
6. The user can view and download the processed video

## MediaPipe Pose Landmarker

The Pose Landmarker model detects 33 landmarks on a human body:
- Face landmarks (nose, eyes, ears)
- Shoulder landmarks
- Elbow and wrist landmarks
- Hip, knee, and ankle landmarks
- Hand and foot landmarks

The application uses the "heavy" model complexity (model_complexity=2) for better accuracy, although this may be slower than lighter models.

## Troubleshooting

- **Video not uploading**: Ensure your video file is in a supported format (MP4, AVI, MOV, WEBM) and is less than 50MB
- **Processing takes too long**: The processing time depends on the video length and your computer's processing power
- **No landmarks detected**: Ensure there is a person clearly visible in the video frames

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [MediaPipe](https://mediapipe.dev/) - For the Pose Landmarker solution
- [Flask](https://flask.palletsprojects.com/) - For the web framework
- [OpenCV](https://opencv.org/) - For video processing capabilities 