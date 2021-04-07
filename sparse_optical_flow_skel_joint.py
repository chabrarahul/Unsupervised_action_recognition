import cv2
import numpy as np
import json

#jsonfile = "/content/drive/MyDrive/skel_out/alphapose-results.json"
json_file=open("/content/drive/MyDrive/skel_out_tire_new_final/alphapose-results.json",)
print(json_file)
tracks=json.load(json_file)
print(tracks)
lastframe = -1 #name of the first frame
skeletons = []
#bboxes = []
tracklist = [] #storing whether the person is in track in certain frames
isLead = False
for track in (tracks):
    frame = int(track['image_id'][:-4])
    if not isLead:
        if track['idx']!=1:
            continue
        else:
            isLead = True
            LeadStart = frame
            lastframe = frame
    #if lastframe==frame:
    #    continue
    if track['idx']!=1:
        continue
    for i in range(frame-lastframe-1): #number of frames that has been skipped
        skeletons.append(np.zeros((17,3)))
        tracklist.append(0)
        #bboxes.append(0)
    pts = np.array(track['keypoints'])
    skeletons.append(pts)
    tracklist.append(1)
    #bboxes.append(track['box'])
    lastframe = frame

print(len(skeletons))


import numpy as np
import cv2
velocity_vec = []
cap = cv2.VideoCapture('/content/drive/MyDrive/video/tire_final_demo_new.mp4')

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
p0_del = skeletons[0].reshape((17,3))
p0_final = np.delete(p0_del, 2, 1)
p0_final = p0_final.astype(np.float32)
p0 = np.expand_dims(p0_final, axis=1)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
#print(p0_final.shape)
#print(p0.dtype)
frame_count = 1
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    #cv2.imsave('/content/drive/MyDrive/frame_trial_sparse_2', add)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    p1_cut = p1.copy()
    p0_cut = p0.copy()
    #p0_final = np.delete(p0_del, 2, 1)
    sub = np.subtract(p1_cut, p0_cut)
    sub_final = np.squeeze(sub, axis=1)
    print(sub_final.shape)
    velocity_vec.append(sub_final)
    old_gray = frame_gray.copy()
    p0_del = skeletons[frame_count].reshape((17,3))
    p0_final = np.delete(p0_del, 2, 1)
    p0_final = p0_final.astype(np.float32)
    p0 = np.expand_dims(p0_final, axis=1)
    frame_count = frame_count + 1

velocity_vec_final = np.array(velocity_vec)
cv2.destroyAllWindows()
cap.release()
