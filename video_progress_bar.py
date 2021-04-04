import numpy as np
import cv2
import math


def getStats(capture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = capture.get(cv2.CAP_PROP_FOURCC)
    durationSec = frame_count/fps * 1000
    return fps, frame_count, durationSec

def visualize_video(video_file):
    cap = cv2.VideoCapture(video_file)
    fps, frame_count, durationSec = getStats(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{video_file}_annotated_new.avi',fourcc, fps, (800,600))
    print("Total time: {durationSec}s FrameRate: {fps} FrameCount: {frame_count}")
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    colors = [(0,0,0), (0,0,255), (0,255,0), (255,255,0)] 
    #(0,255,255),(255,0,255),
    #         (144,255,0), (87,22,8), (255,111,9), (100,100,100), (0,0,0)]
    
    thickness = 2
    frame_no = 0
    action_idx = 0
    transcript = [0,1,2,3]
    transcript_name =  ['action_1', 'action_2', 'action_3', 'action_4']
    while(True):
        ret, frame = cap.read()
        if ret:
            # Capture frame-by-frame
            # Our operations on the frame come here
            frame = cv2.resize(frame, (800, 600))
            index = data[frame_no]
            text_action = transcript_name[index]
            # Using cv2.putText() method
            frame = cv2.putText(frame, text_action, org, font,
                            fontScale, colors[index], thickness, cv2.LINE_AA)
            draw_bar = 0
            for i in data:
                start = int(draw_bar/frame_count * 600) + 100
                end = int(draw_bar/frame_count * 600) + 100
                frame = cv2.line(frame, (start, 550), (end, 550), colors[i], 5)
                draw_bar += 1

            start = int(frame_no/frame_count * 600) + 100
            end = int(frame_no/frame_count * 600) + 100
            frame = cv2.line(frame, (start, 540), (end, 560), (255,255,255), 4)
            out.write(frame)
            # Display the resulting frame
            #cv2.imwrite('/content/drive/MyDrive/frame_trial_new.png', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
                # break
            frame_no += 1
        else:
            break
    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    #transcript = read_transcript('worker_B.txt')
    visualize_video('/content/drive/MyDrive/rahul.mp4')
