import cv2

#april = 'tag21_07'
april = 'tag36_11'
size = 750
for i in range(4):
    filename = '{}_{:05}.png'.format(april, i)
    print('Load:', filename)
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img[img[:,:,3]==0] = 255
    img = img[:,:,0]
    # calculate factor
    factor = size/max(img.shape[0],img.shape[1])
    img = cv2.resize(img, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
    # save image
    savefilename = '{}_{:05}_{}.png'.format(april, i, size)
    cv2.imwrite(savefilename, img)
print('End')
