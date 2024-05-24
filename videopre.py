# import cv2
# import os

# from my_clip import generate_caption
# from combining import generate_combined_sentence_with_bart
# from cnnRnn import generate_image_description


# def extract_frames(video_path, frame_interval=200):
#     output_dir = 'static/uploads/frames101'
#     """
#     Extract frames from a video at a specified interval and save them as images.
#     Generate captions for each extracted frame.

#     Args:
#     - video_path (str): Path to the video file.
#     - output_dir (str): Directory where the extracted frames will be saved.
#     - frame_interval (int): Interval (in frames) at which to extract frames. Default is 200.

#     Returns:
#     - List of dictionaries containing 'image_path' and 'caption'.
#     """
#     video_capture = cv2.VideoCapture(video_path)

#     # Check if the video file was opened successfully
#     if not video_capture.isOpened():
#         print('Error opening video file')
#         return []

#     # Get the frame rate of the video
#     frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

#     # Create the output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Initialize list to store image paths and captions
#     result_data = []

#     # Iterate over the video frames and save them as images
#     frame_count = 0
#     while video_capture.isOpened():
#         ret, frame = video_capture.read()

#         # Check if the frame was read successfully
#         if not ret:
#             break

#         # Save the frame as an image at the specified interval
#         if frame_count % frame_interval == 0:
#             image_name = os.path.join(output_dir, f'frame{frame_count}.jpg')
#             image_nma = image_name.removeprefix('static/uploads/frames101')
#             image_nma= image_nma[1:]

#             print(image_nma+" "+image_name)
#             cv2.imwrite(image_name, frame)

#             # Generate caption for the extracted frame (Assuming generate_caption() is defined elsewhere)
#             caption = generate_caption(image_name)
#             caption1= generate_image_description(image_name)
#             result = generate_combined_sentence_with_bart(caption, caption1)
            
#             # Store image path and caption in the result_data list
#             result_data.append({'image_path': image_nma, 'caption1': caption, 'caption2':caption1,'caption3': result})

#         # Increment the frame count
#         frame_count += 1

#     # Release the video capture object
#     video_capture.release()

#     # Close all windows
#     cv2.destroyAllWindows()

#     return result_data

# # Example usage:
# # extracted_data = extract_frames("C:\\Users\\Vaishanavi\\Downloads\\pexels_videos_1576365 (720p).mp4")
import cv2
import os
from gtts import gTTS
from pydub import AudioSegment

from my_clip import generate_caption
from combining import generate_new_sentence
from cnnRnn import generate_image_description

def extract_frames(video_path, frame_interval=100):
    output_dir = 'static/uploads/frames101'
    audio_output_dir = 'static/uploads'
    
    # Create the output directories if they don't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(audio_output_dir):
        os.makedirs(audio_output_dir)

    video_capture = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not video_capture.isOpened():
        print('Error opening video file')
        return []

    # Get the frame rate of the video
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

    # Initialize lists to store image paths and captions
    result_data = []
    all_captions = []  # List to store captions for all frames

    # Iterate over the video frames and save them as images
    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        # Check if the frame was read successfully
        if not ret:
            break

        # Save the frame as an image at the specified interval
        if frame_count % frame_interval == 0:
            image_name = os.path.join(output_dir, f'frame{frame_count}.jpg')
            image_nma = image_name.removeprefix('static/uploads/frames101')
            image_nma= image_nma[1:] 
            cv2.imwrite(image_name, frame)

            # Generate captions for the extracted frame
            caption = generate_caption(image_name)
            caption1 = generate_image_description(image_name)
            combined_caption = generate_new_sentence(caption, caption1)

            result_data.append({'image_path': image_nma, 'caption1': caption, 'caption2': caption1, 'caption3': combined_caption,'audio' :'combined_audio.mp3'})

            # Append current caption to the list of all captions
            all_captions.append(combined_caption)

        # Increment the frame count
        frame_count += 1

    # Combine all captions into one sentence
    combined_sentence = ' '.join(all_captions)

    # Convert combined sentence to audio
    tts = gTTS(text=combined_sentence, lang='en')
    combined_audio_path = os.path.join(audio_output_dir, 'combined_audio.mp3')
    tts.save(combined_audio_path)

    # Release the video capture object
    video_capture.release()

    return result_data

# # Example usage:
# extracted_data, combined_audio_path = extract_frames("C:\\Users\\Vaishanavi\\Downloads\\pexels_videos_1576365 (720p).mp4")
