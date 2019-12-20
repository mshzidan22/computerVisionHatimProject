from utils import *
import subprocess
from matplotlib import pyplot as plt



max_val = 20
max_pt = -1
max_kp = 0

orb = cv2.ORB_create()
# orb is an alternative to SIFT.Sift is not working I don't know why?

test_img = read_img('test/5-1.jpg')


# resizing and showing the test image
original = resize_img(test_img,0.8)
display('original', original)

# keypoints and descriptors
(kp1, des1) = orb.detectAndCompute(test_img, None)

training_set = ['currencyDataSet/1-1.jpg', 'currencyDataSet/1-2.jpg', 'currencyDataSet/5-1.jpg', 'currencyDataSet/5-2.jpg','currencyDataSet/5-3.jpg','currencyDataSet/10-1.jpg','currencyDataSet/10-2.jpg','currencyDataSet/20-1.jpg','currencyDataSet/20-2.jpg','currencyDataSet/50-1.jpg','currencyDataSet/50-2.jpg','currencyDataSet/100-1.jpg','currencyDataSet/100-2.jpg','currencyDataSet/200-1.jpg','currencyDataSet/200-2.jpg']

for i in range(0, len(training_set)):
    # train image
    train_img = cv2.imread(training_set[i])

    (kp2, des2) = orb.detectAndCompute(train_img, None)

    # brute force matcher for matching discriptors
    bf= cv2.BFMatcher(cv2.NORM_HAMMING)
    all_matches = bf.knnMatch(des1, des2, k=2)
    #Lowe's ratio test
    # this test checks if matches are ambiguous and should be removed.
    good = []
    # give an arbitrary number -> 0.789
    # if good -> append to list of good matches
    for (m, n) in all_matches:
        if m.distance < 0.81 * n.distance:
            good.append([m])
     
    if len(good) > max_val:
        max_val = len(good)
        max_pt = i
        max_kp = kp2

    print(i, ' ', training_set[i], ' have ', len(good),' acepted matches')

if max_val != 20:
    print("The best image is ",training_set[max_pt])
    print('Have good matches ', max_val)
    note = str(training_set[max_pt])[training_set[max_pt].index('/')+1:training_set[max_pt].index('-')]
    print('\n the currency is ====> ', note ,'pound')
    
    #draw the the images
    #train_img = cv2.imread(training_set[max_pt])
    #img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
    #plt.imshow(img3)
    #plt.show()
else:
    print('Fake currency')