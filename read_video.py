def read_video(video_filename, width=224, height=224):
    cap = cv2.VideoCapture(video_filename)
    frames = []
    if cap.isOpened():
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (width, height))
            frames.append(frame_rgb)
    frames = np.asarray(frames)
    return frames


def load_videos(path_to_raw_videos):
    drive.mount('/content/gdrive')
    video_filenames = sorted(glob.glob(os.path.join(path_to_raw_videos, '*.mp4')))
    print('Found %d videos to align.'%len(video_filenames))
    videos = []
    video_seq_lens = []
    
    for video_filename in video_filenames:
        frames = read_video(video_filename)
        videos.append(frames)
        video_seq_lens.append(len(frames))
  
  max_seq_len = max(video_seq_lens)
  videos = np.asarray([pad_zeros(x, max_seq_len) for x in videos])
  return videos, video_seq_lens
