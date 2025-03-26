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
import sys

# Set page config
st.set_page_config(
    page_title="MediaPipe Pose Landmarker",
    page_icon="🧍",
    layout="centered"
)

# Create a dedicated writable directory for MediaPipe
TEMP_DIR = tempfile.gettempdir()
MODEL_DIR = os.path.join(TEMP_DIR, "mediapipe_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Set MediaPipe environment variables
os.environ["MEDIAPIPE_MODEL_PATH"] = MODEL_DIR
os.environ["MEDIAPIPE_RESOURCE_DIR"] = MODEL_DIR
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

# Monkey patch MediaPipe's download functionality to save models to our writable directory
def patch_mediapipe_download():
    try:
        from mediapipe.python.solutions import download_utils
        original_download = download_utils.download_oss_model
        
        def patched_download(model_path, file_name):
            # Redirect to our writable model directory
            custom_model_path = os.path.join(MODEL_DIR, file_name)
            
            try:
                # Return early if the model already exists
                if os.path.exists(custom_model_path):
                    return custom_model_path
                
                # Create any necessary subdirectories
                os.makedirs(os.path.dirname(custom_model_path), exist_ok=True)
                
                # Get the model URL but save to our custom path
                model_url = f'https://storage.googleapis.com/mediapipe-assets/{model_path}'
                st.info(f"Downloading model from {model_url} to {custom_model_path}")
                
                import urllib.request
                with urllib.request.urlopen(model_url) as response, open(custom_model_path, 'wb') as f:
                    shutil.copyfileobj(response, f)
                    
                return custom_model_path
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                raise e
        
        # Replace the original function with our patched version
        download_utils.download_oss_model = patched_download
        return True
    except Exception as e:
        st.error(f"Failed to patch MediaPipe download: {e}")
        return False

# Function to get accurate video dimensions using FFmpeg
def get_video_info(video_path):
    try:
        # Check if ffprobe is available
        try:
            result = subprocess.run(['ffprobe', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                return None, None, None
        except:
            return None, None, None
            
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
        # Patch MediaPipe download functionality
        patch_success = patch_mediapipe_download()
        if not patch_success:
            st.warning("Could not patch MediaPipe download function. Will try direct initialization.")
        
        # Load MediaPipe components with light model (less likely to cause issues)
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Use medium model instead of heavy
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        return pose, mp_pose, mp.solutions.drawing_utils, mp.solutions.drawing_styles
    except Exception as e:
        # Try fallback to even lighter model if medium fails
        try:
            st.warning(f"Error loading medium model: {e}. Trying lighter model...")
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # Use lightest model
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            return pose, mp_pose, mp.solutions.drawing_utils, mp.solutions.drawing_styles
        except Exception as e2:
            st.error(f"Error loading MediaPipe: {e2}")
            st.error("MediaPipe initialization failed. This app may not work correctly in this environment.")
            st.info("If this error persists, please try running the app locally instead of on Streamlit Cloud.")
            # Return None values to allow the app to continue without MediaPipe functionality
            return None, None, None, None

# App header
st.title("MediaPipe Pose Landmarker")
st.markdown("Upload a video to detect and visualize pose landmarks using MediaPipe's model.")

# Display Python and environment information
with st.expander("Environment Information"):
    st.code(f"""
Python version: {sys.version}
Temp directory: {TEMP_DIR}
Model directory: {MODEL_DIR}
Directory exists: {os.path.exists(MODEL_DIR)}
Directory writable: {os.access(MODEL_DIR, os.W_OK)}
Current working directory: {os.getcwd()}
    """)

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
    
    # Speed control option
    st.write("Select video playback speed:")
    speed_options = {
        "Normal (1.0x)": 1.0,
        "Slow (0.75x)": 0.75,
        "Slower (0.5x)": 0.5,
        "Slowest (0.25x)": 0.25
    }
    selected_speed = st.selectbox(
        "Playback Speed",
        options=list(speed_options.keys()),
        index=0
    )
    speed_factor = speed_options[selected_speed]
    
    if speed_factor != 1.0:
        st.info(f"Video will be processed at {speed_factor}x speed. This will help with tracking fast movements.")
    
    # Process button
    if st.button("Process Video"):
        try:
            # Load MediaPipe
            pose, mp_pose, mp_drawing, mp_drawing_styles = load_mediapipe()
            
            # Check if MediaPipe loaded correctly
            if pose is None:
                st.error("MediaPipe components failed to load. Cannot process video with pose detection.")
                st.stop()
            
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
            original_fps = orig_fps if orig_fps and orig_fps > 0 else cap.get(cv2.CAP_PROP_FPS)
            
            # Apply speed factor to FPS for output
            fps = original_fps * speed_factor
            st.info(f"Original FPS: {original_fps:.2f}, Output FPS: {fps:.2f}")
            
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
            
            # Frame skipping variables for slow motion
            frame_indices = []
            if speed_factor == 1.0:
                # Normal speed: use all frames
                while frame_count < total_frames:
                    frame_indices.append(frame_count)
                    frame_count += 1
            else:
                # For slow motion: duplicate frames
                target_frame_count = int(total_frames / speed_factor)
                for i in range(target_frame_count):
                    # Calculate which original frame to use
                    original_frame_idx = min(int(i * speed_factor), total_frames - 1)
                    frame_indices.append(original_frame_idx)
            
            # Reset frame_count for actual processing
            frame_count = 0
            processed_count = 0
            
            # Skip to beginning of video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Dictionary to store frames for slow motion
            frames_cache = {}
            
            # Process all required frames
            for idx, frame_idx in enumerate(frame_indices):
                # If we've already processed this frame, use it from cache
                if frame_idx in frames_cache:
                    # Save the cached frame with new index
                    frame_filename = os.path.join(frames_dir, f"frame_{processed_count:06d}.png")
                    cv2.imwrite(frame_filename, frames_cache[frame_idx])
                    processed_frames.append(frame_filename)
                    processed_count += 1
                    
                    # Update progress
                    progress = (idx + 1) / len(frame_indices)
                    progress_bar.progress(progress)
                    progress_text.text(f"Processing frame {idx+1}/{len(frame_indices)} (reusing frame {frame_idx})")
                    continue
                
                # Seek to the correct frame if not the next one
                if frame_idx != frame_count:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    frame_count = frame_idx
                
                # Read the frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                processed_count += 1
                
                # Update progress
                progress = (idx + 1) / len(frame_indices)
                progress_bar.progress(progress)
                progress_text.text(f"Processing frame {idx+1}/{len(frame_indices)} (original frame {frame_idx})")
                
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
                
                # Initialize new_w and new_h variables
                new_w = orig_width
                new_h = orig_height
                
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
                frame_filename = os.path.join(frames_dir, f"frame_{idx:06d}.png")
                cv2.imwrite(frame_filename, canvas)
                processed_frames.append(frame_filename)
                
                # Store in cache for potential reuse (for slow motion)
                if speed_factor < 1.0:
                    frames_cache[frame_idx] = canvas.copy()
            
            # Release resources
            cap.release()
            
            # Check if we processed any frames
            if not processed_frames:
                st.error("No frames were successfully processed")
                st.stop()
                
            # Combine frames back into a video
            output_path = f"processed_{int(time.time())}.mp4"
            temp_output_path = os.path.join(tempfile.gettempdir(), output_path)
            
            # Use OpenCV to create video - since ffmpeg is not available
            progress_text.text("Combining frames into video...")
            out = cv2.VideoWriter(
                temp_output_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                fps,  # Use the speed-adjusted fps 
                (orig_width, orig_height)
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
            out_fps = check_cap.get(cv2.CAP_PROP_FPS)
            check_cap.release()
            
            st.info(f"Processed video dimensions: {out_width}×{out_height}, FPS: {out_fps:.2f}")
            
            # Check if dimensions exactly match
            if out_width != orig_width or out_height != orig_height:
                st.warning(f"Final dimensions ({out_width}×{out_height}) don't exactly match target dimensions ({orig_width}×{orig_height}).")
                st.info("This is likely due to pixel alignment requirements for video encoding, but the difference should be negligible (1-2 pixels).")
            
            # Clear progress indicators
            progress_text.empty()
            progress_bar.empty()
            
            # Display success message
            if speed_factor < 1.0:
                st.success(f"Video processing complete! Video has been slowed to {speed_factor}x speed.")
            else:
                st.success("Video processing complete!")
            
            # Check if output file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                # Display the processed video
                st.video(output_path)
                
                # Download button
                with open(output_path, 'rb') as file:
                    download_filename = f"processed_pose_video_{speed_factor}x.mp4" if speed_factor < 1.0 else "processed_pose_video.mp4"
                    st.download_button(
                        label="Download Processed Video",
                        data=file,
                        file_name=download_filename,
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
    This application uses MediaPipe's Pose Landmarker to detect and visualize pose landmarks in videos.
    
    The Pose Landmarker model detects 33 landmarks on a human body:
    - Face landmarks (nose, eyes, ears)
    - Shoulder landmarks
    - Elbow and wrist landmarks
    - Hip, knee, and ankle landmarks
    - Hand and foot landmarks
    
    The application tries to use the best model available in the current environment.
    
    You can also slow down videos to better analyze fast movements.
    """)