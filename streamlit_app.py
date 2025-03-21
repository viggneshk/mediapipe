import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os
import time
import subprocess
import json
import numpy as np
import shutil

# Set page config
st.set_page_config(
    page_title="MediaPipe Pose Landmarker",
    page_icon="🧍",
    layout="centered"
)

# Set environment variable to specify a writable directory for MediaPipe models
os.environ["MEDIAPIPE_MODEL_PATH"] = tempfile.gettempdir()

# Function to get accurate video dimensions using FFmpeg
def get_video_info(video_path):
    try:
        # Run ffprobe to get video information in JSON format
        cmd = [
            'ffprobe', 
            '-v', 'quiet', 
            '-print_format', 'json', 
            '-show_format', 
            '-show_streams', 
            video_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        info = json.loads(result.stdout)
        
        # Find the video stream
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                width = int(stream.get('width', 0))
                height = int(stream.get('height', 0))
                
                # Sometimes rotation needs to be considered
                rotation = int(stream.get('tags', {}).get('rotate', '0'))
                if rotation in [90, 270]:
                    width, height = height, width
                    
                fps_parts = stream.get('r_frame_rate', '30/1').split('/')
                fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
                
                # Some mobile videos have display aspect ratio different from storage
                if 'display_aspect_ratio' in stream:
                    try:
                        display_ratio = stream['display_aspect_ratio']
                        if ':' in display_ratio:
                            num, den = display_ratio.split(':')
                            display_ratio = float(num) / float(den)
                            calculated_width = int(height * display_ratio)
                            if abs(calculated_width - width) > 10:  # If significant difference
                                width = calculated_width
                    except:
                        pass
                
                return width, height, fps
        
        # Fallback to OpenCV if no video stream found
        return None, None, None
    except Exception as e:
        st.warning(f"Error getting video info with FFmpeg: {e}")
        return None, None, None

# Alternative method to get accurate dimensions by reading a frame
def get_dimensions_from_frame(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None
            
        # Read the first frame
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None, None
            
        # Get dimensions from the actual frame
        height, width = frame.shape[:2]
        cap.release()
        return width, height
    except:
        return None, None

# Initialize MediaPipe Pose
@st.cache_resource
def load_mediapipe():
    try:
        # Create a temp dir for MediaPipe models if needed
        model_dir = os.path.join(tempfile.gettempdir(), "mediapipe_models")
        os.makedirs(model_dir, exist_ok=True)
        
        # Set environment variables for MediaPipe
        os.environ["MEDIAPIPE_RESOURCE_DIR"] = model_dir
        
        # Load MediaPipe components
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 2 = heavy model
            enable_segmentation=False,  # Disable segmentation to avoid the error
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        return pose, mp_pose, mp.solutions.drawing_utils, mp.solutions.drawing_styles
    except Exception as e:
        st.error(f"Error loading MediaPipe: {e}")
        st.info("If this error persists, please try a different video or check if MediaPipe is compatible with your system.")
        raise e

# App header
st.title("MediaPipe Pose Landmarker")
st.markdown("Upload a video to detect and visualize pose landmarks using MediaPipe's heavy model.")

# File uploader
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "webm"])

if uploaded_file is not None:
    # Create a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    input_path = tfile.name
    
    # Try multiple methods to get accurate dimensions
    orig_width, orig_height, orig_fps = get_video_info(input_path)
    
    # Try getting dimensions from first frame if ffprobe didn't work well
    if orig_width is None or orig_height is None or orig_width <= 0 or orig_height <= 0:
        frame_width, frame_height = get_dimensions_from_frame(input_path)
        if frame_width and frame_height:
            orig_width, orig_height = frame_width, frame_height
    
    # Fallback to OpenCV metadata if other methods fail
    if orig_width is None or orig_height is None or orig_width <= 0 or orig_height <= 0:
        cap_check = cv2.VideoCapture(input_path)
        orig_width = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap_check.get(cv2.CAP_PROP_FPS)
        cap_check.release()
    
    # Display detected dimensions
    st.info(f"Detected video dimensions: {orig_width}×{orig_height}, FPS: {orig_fps if orig_fps else 'unknown'}")
    
    # Allow manual dimension override if needed
    st.write("If the detected dimensions are incorrect, you can specify them manually:")
    
    col1, col2 = st.columns(2)
    with col1:
        manual_width = st.number_input("Width", value=orig_width, min_value=1, step=1)
    with col2:
        manual_height = st.number_input("Height", value=orig_height, min_value=1, step=1)
    
    # Use manual dimensions if changed
    if manual_width != orig_width or manual_height != orig_height:
        orig_width = manual_width
        orig_height = manual_height
        st.info(f"Using manual dimensions: {orig_width}×{orig_height}")
    
    # Process button
    if st.button("Process Video"):
        try:
            # Load MediaPipe
            pose, mp_pose, mp_drawing, mp_drawing_styles = load_mediapipe()
            
            # Progress placeholder
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Open video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error("Error opening video file")
                st.stop()
            
            # Get video properties as read by OpenCV (may differ from actual)
            cv_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cv_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = orig_fps if orig_fps and orig_fps > 0 else cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if cv_width != orig_width or cv_height != orig_height:
                st.warning(f"OpenCV reports different dimensions ({cv_width}×{cv_height}) " +
                          f"than the target dimensions ({orig_width}×{orig_height}).")
                
                # Calculate aspect ratio from source dimensions
                source_aspect_ratio = cv_width / cv_height
                target_aspect_ratio = orig_width / orig_height
                
                if abs(source_aspect_ratio - target_aspect_ratio) > 0.01:  # More than 1% difference
                    st.warning(f"The aspect ratios differ significantly: source ({source_aspect_ratio:.3f}) vs target ({target_aspect_ratio:.3f}). " +
                              f"This might cause the video to appear stretched or squeezed.")
            
            # Create a directory for processed frames (with target dimensions)
            frames_dir = tempfile.mkdtemp()
            processed_frames = []
            
            # Process video frame by frame
            frame_count = 0
            
            # Calculate optimal processing dimensions to preserve aspect ratio
            source_aspect = cv_width / cv_height
            target_aspect = orig_width / orig_height
            
            # Force preserving the source aspect ratio
            # If we're using the target width and height directly (manual override)
            # we need to ensure we don't stretch/squeeze the content
            if abs(source_aspect - target_aspect) > 0.01:  # If aspect ratios differ by more than 1%
                st.info("Using letterboxing/pillarboxing to preserve original aspect ratio")
                preserve_source_aspect = True
            else:
                preserve_source_aspect = False
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                progress_text.text(f"Processing frame {frame_count}/{total_frames}")
                
                # Convert the BGR image to RGB for MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # Process the frame with MediaPipe Pose
                    results = pose.process(image_rgb)
                    
                    # Draw pose landmarks on the original frame
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                except Exception as frame_error:
                    st.warning(f"Error processing frame {frame_count}: {frame_error}")
                    # Continue with the next frame
                
                # Create a black canvas of the target dimensions
                canvas = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)
                
                # If preserving source aspect ratio:
                if preserve_source_aspect:
                    # Calculate the dimensions that preserve the original aspect ratio
                    # while fitting within the target dimensions
                    if source_aspect > target_aspect:  # Source is wider, letterbox (black bars on top/bottom)
                        new_w = orig_width
                        new_h = int(new_w / source_aspect)
                    else:  # Source is taller, pillarbox (black bars on sides)
                        new_h = orig_height
                        new_w = int(new_h * source_aspect)
                    
                    # Resize the frame to the calculated dimensions
                    resized = cv2.resize(frame, (new_w, new_h))
                    
                    # Calculate position to center the image
                    x_offset = (orig_width - new_w) // 2
                    y_offset = (orig_height - new_h) // 2
                else:
                    # Standard resize to target dimensions
                    resized = cv2.resize(frame, (orig_width, orig_height))
                    x_offset = 0
                    y_offset = 0
                
                # Ensure valid dimensions for placement
                if y_offset < 0: y_offset = 0
                if x_offset < 0: x_offset = 0
                if y_offset + new_h > orig_height: new_h = orig_height - y_offset
                if x_offset + new_w > orig_width: new_w = orig_width - x_offset
                
                # Place the resized image on the canvas
                if preserve_source_aspect:
                    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                else:
                    canvas = resized
                
                # Save the frame with the exact target dimensions
                frame_filename = os.path.join(frames_dir, f"frame_{frame_count:06d}.png")
                cv2.imwrite(frame_filename, canvas)
                processed_frames.append(frame_filename)
            
            # Release resources
            cap.release()
            
            # Check if we processed any frames
            if not processed_frames:
                st.error("No frames were successfully processed")
                st.stop()
                
            # Combine frames back into a video
            output_path = f"processed_{int(time.time())}.mp4"
            temp_output_path = os.path.join(tempfile.gettempdir(), output_path)
            
            # Use FFmpeg to combine frames directly (since they're already at target dimensions)
            try:
                command = [
                    'ffmpeg',
                    '-y',  # Overwrite output file if it exists
                    '-framerate', str(fps),
                    '-i', os.path.join(frames_dir, 'frame_%06d.png'),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    temp_output_path
                ]
                
                process = subprocess.Popen(
                    command, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    st.error(f"FFmpeg error: {stderr.decode()}")
                    # Fallback to OpenCV
                    progress_text.text("FFmpeg failed, falling back to OpenCV...")
                    out = cv2.VideoWriter(
                        temp_output_path, 
                        cv2.VideoWriter_fourcc(*'mp4v'), 
                        fps, 
                        (orig_width, orig_height)  # Use target dimensions
                    )
                    
                    for frame_file in processed_frames:
                        frame = cv2.imread(frame_file)
                        out.write(frame)
                    
                    out.release()
            except Exception as e:
                st.error(f"Error using FFmpeg: {e}")
                
                # Fallback to OpenCV
                progress_text.text("FFmpeg failed, falling back to OpenCV...")
                out = cv2.VideoWriter(
                    temp_output_path, 
                    cv2.VideoWriter_fourcc(*'mp4v'), 
                    fps, 
                    (orig_width, orig_height)  # Use target dimensions
                )
                
                for frame_file in processed_frames:
                    frame = cv2.imread(frame_file)
                    out.write(frame)
                
                out.release()
            
            # Copy to a location accessible by Streamlit
            if os.path.exists(temp_output_path):
                with open(temp_output_path, 'rb') as f_in:
                    video_data = f_in.read()
                    
                # Write to current directory
                with open(output_path, 'wb') as f_out:
                    f_out.write(video_data)
            
            # Verify output video dimensions
            check_cap = cv2.VideoCapture(output_path)
            out_width = int(check_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            out_height = int(check_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            check_cap.release()
            
            st.info(f"Processed video dimensions: {out_width}×{out_height}")
            
            # Check if dimensions exactly match
            if out_width != orig_width or out_height != orig_height:
                st.warning(f"Final dimensions ({out_width}×{out_height}) don't exactly match target dimensions ({orig_width}×{orig_height}).")
                st.info("This is likely due to pixel alignment requirements for video encoding, but the difference should be negligible (1-2 pixels).")
            
            # Clear progress indicators
            progress_text.empty()
            progress_bar.empty()
            
            # Display success message
            st.success("Video processing complete!")
            
            # Check if output file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                # Display the processed video
                st.video(output_path)
                
                # Download button
                with open(output_path, 'rb') as file:
                    st.download_button(
                        label="Download Processed Video",
                        data=file,
                        file_name=f"processed_pose_video.mp4",
                        mime="video/mp4"
                    )
            else:
                st.error("Output video file is missing or empty")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.error(traceback.format_exc())
            
        finally:
            # Clean up temporary files
            if os.path.exists(input_path):
                os.unlink(input_path)
            
            # Clean up frames directory
            if 'frames_dir' in locals() and os.path.exists(frames_dir):
                for frame_file in processed_frames:
                    if os.path.exists(frame_file):
                        try:
                            os.unlink(frame_file)
                        except:
                            pass
                try:
                    os.rmdir(frames_dir)
                except:
                    pass
                
            # Clean up temp output file
            if 'temp_output_path' in locals() and os.path.exists(temp_output_path):
                try:
                    os.unlink(temp_output_path)
                except:
                    pass
                
# Display info about the model
with st.expander("About MediaPipe Pose Landmarker"):
    st.markdown("""
    This application uses MediaPipe's Pose Landmarker with the heavy model to detect and visualize pose landmarks in videos.
    
    The Pose Landmarker model detects 33 landmarks on a human body:
    - Face landmarks (nose, eyes, ears)
    - Shoulder landmarks
    - Elbow and wrist landmarks
    - Hip, knee, and ankle landmarks
    - Hand and foot landmarks
    
    The heavy model provides the most accurate landmark detection but may be slower than lighter models.
    """)