import numpy as np, pylab as pyl

#orientation rules wrt to chip 50
# rotate clockwise 180



def make_display_image(frame, step=9):
    (A, B, C) = frame.shape

    for i in range(0,int(A/3),step):
        if i ==0:
            im = frame[i, :, :].T
        else:
            im = np.concatenate([im,frame[i, :, :].T])
    return im.T


def shuffle(frame, shuffleFactor=2, n_shuffle = 1):
    """
    shuffle around the image by small amount

    Frame is a 3d frame of shape (C, B, A, A, 1)

    the shuffling will be +/- 1/expandFactor pixels in both x and y

    """

    (C, B, A, junk, junker) = frame.shape
    shuffled_frames = np.zeros(frame.shape,dtype=frame.dtype)
    for j in range(n_shuffle):
        for i in range(A):
            (dx,dy) = np.random.randint(low=-shuffleFactor, high=shuffleFactor, size=(2))
            #print(dx,dy)

            #shuffled_frames[j].append(np.roll(np.roll(frame,dx,axis=2), axis=3))
            #fig = pyl.figure(1)
            #sp1 = fig.add_subplot(1,2,1)
            #pyl.imshow(frame[i,5,:,:,0])
            shuffled_frames[i:i+1,:,:,:,:]=np.roll(frame[i:i+1,:,:,:,:],shift=(dx,dy), axis=(2,3))
            #sp2 = fig.add_subplot(1,2,2)
            #pyl.imshow(shuffled_frames[i,5,:,:,0])
            #pyl.show()
    return np.array(shuffled_frames)
