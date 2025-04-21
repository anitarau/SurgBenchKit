 # added by Anita Rau April 2025


def extract_heichole_frames(video_folder, output_folder):
    """
    Extracts frames from all .mp4 videos in the specified video folder and saves
    them in corresponding subdirectories within the output folder.

    Parameters:
    video_folder (str): The path to the folder containing .mp4 video files.
    output_folder (str): The path to the folder where extracted frames will be saved.
    """
    import os
    import subprocess       
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all video files in the video folder
    videos = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    #videos = [f for f in videos if int(f.split('Chole')[1].split('.')[0]) in [6, 19, 10, 8, 20]] # val seq
    videos = [f for f in videos if int(f.split('Chole')[1].split('.')[0]) in [4, 1, 22, 16, 13]] # test seqs

    # Loop through each video file
    for video in videos:
        # Extract the base name of the video without the extension
        base_name = os.path.splitext(video)[0]
        
        # Create a directory for the extracted frames for each video
        output_path = os.path.join(output_folder, base_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Construct the FFmpeg command
        command = [
            'ffmpeg',
            '-i', os.path.join(video_folder, video),
            '-start_number', '0',
            os.path.join(output_path, 'frame_%05d.png')
        ]
        
        # Run the command
        subprocess.run(command)
