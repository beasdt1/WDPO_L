import cv2
import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
# some needed shit
def load_image(filename='', flag=cv2.IMREAD_COLOR, scale=1.0):
    path = 'C:/Users/molsz/OneDrive/PUT/WDPO/data/img/'+filename
    tmp = cv2.imread(path, flag)
    if tmp is None:
        sys.exit("Could not read the image"+filename)
    else:
        tmp = cv2.resize(tmp, (0, 0), fx=scale, fy=scale)
        return tmp

def empty_callback(value):
    #print(f'Trackbar reporting for duty with value: {value}')
    pass

# LAB01
def lab01():
    def cam():
        cap = cv2.VideoCapture(0)  # open the default camera

        key = ord('a')
        while key != ord('q'):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame comes here
            # Convert RGB image to grayscale
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Blur the image
            img_filtered = cv2.GaussianBlur(img_gray, (7, 7), 1.5)
            # Detect edges on the blurred image
            img_edges = cv2.Canny(img_filtered, 0, 30, 3)

            # Display the result of our processing
            cv2.imshow('result', img_edges)
            # Wait a little (30 ms) for a key press - this is required to refresh the image in our window
            key = cv2.waitKey(30)

        # When everything done, release the capture
        cap.release()
        # and destroy created windows, so that they are not left for the rest of the program
        cv2.destroyAllWindows()

    def ork_image():
        ork_img = load_image('pobranyork.jpeg')
        cv2.imshow("ork", ork_img)
        cv2.waitKey(0)
        cv2.imwrite("ork.jpg", ork_img)
        print(ork_img.shape)
        px = ork_img[150, 150]
        print(f'Pixel value at [150, 150]: {px}')

    def ork_image_grey():
        ork_img_grey = load_image('pobranyork.jpeg', cv2.IMREAD_GRAYSCALE)
        cv2.imshow("ork", ork_img_grey)
        cv2.waitKey(0)
        print(ork_img_grey.shape)
        px = ork_img_grey[150, 150]
        print(f'Pixel value at [150, 150]: {px}')

    def duplicate_roi():
        img_strefa = load_image('strefaruchu.jpg')
        cv2.imshow("strefa ruchu", img_strefa)
        cv2.waitKey(0)
        img_sign = img_strefa[0:150, 170:580]
        cv2.imshow("znak", img_sign)
        cv2.waitKey(0)
        img_strefa[150:300, 170:580] = img_sign
        cv2.imshow("2x strefa ruchu", img_strefa)
        cv2.waitKey(0)

    def split_image():
        img_rgb = load_image('AdditiveColor.png')
        cv2.imshow("obrazek RGB", img_rgb)
        cv2.waitKey(0)
        b, g, r = cv2.split(img_rgb)
        cv2.imshow("Blue component", b)
        cv2.imshow("Green component", g)
        cv2.imshow("Red component", r)
        cv2.waitKey(0)

    def camera_frame():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Display the resulting frame
            cv2.imshow('frame', gray)
            button = cv2.waitKey(0)
            if button == ord('q'):
                break
            elif button == 32:
                continue
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    def video_playback_framebyframe():
        path = 'C:/Users/molsz/OneDrive/PUT/WDPO/data/video/wildlife.mp4'
        vid = cv2.VideoCapture(path)
        if not vid.isOpened():
            print("cannot open file wildlife.mp4")
        while True:
            ret, frame = vid.read()
            if ret:
                cv2.imshow('wildlife', frame)
                button = cv2.waitKey(0)
                if button == ord('q'):
                    break
                elif button == ord('e'):
                    continue
            else:
                print("end of file")
                break
        # When everything done, release the capture
        vid.release()
        cv2.destroyAllWindows()

    def gallery_qe():
        path = 'C:/Users/molsz/OneDrive/PUT/WDPO/data/img/'
        imgs = []
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                imgs.append(path + file)
        i = 0
        while True:
            img = cv2.imread(imgs[i])
            cv2.imshow(f'obrazek{i}', img)
            k = cv2.waitKey(0)
            if k == ord('e'):
                i = i + 1
                if i == len(imgs):
                    i = 0
            elif k == ord('q'):
                i = i - 1
                if i == -1:
                    i = len(imgs) - 1
            elif k == 32:
                break
            cv2.destroyAllWindows()

    # wywolania

    # cam()
    # ork_image()
    # ork_image_grey()
    # duplicate_roi()
    # split_image()
    # camera_frame()
    # video_playback_framebyframe()
    gallery_qe()

# LAB02
def lab02():
    def trackbar_test():
        # create a black image, a window
        img = np.zeros((300, 700, 3), dtype=np.uint8)
        cv2.namedWindow('image')

        # create trackbars for color change
        cv2.createTrackbar('R', 'image', 0, 255, empty_callback)
        cv2.createTrackbar('G', 'image', 0, 255, empty_callback)
        cv2.createTrackbar('B', 'image', 0, 255, empty_callback)

        # create switch for ON/OFF functionality
        switch_trackbar_name = '0 : OFF \n1 : ON'
        cv2.createTrackbar(switch_trackbar_name, 'image', 0, 1, empty_callback)

        while True:
            cv2.imshow('image', img)

            # sleep for 10 ms waiting for user to press some key, return -1 on timeout
            key_code = cv2.waitKey(10)
            if key_code == 27:
                # escape key pressed
                break

            # get current positions of four trackbars
            r = cv2.getTrackbarPos('R', 'image')
            g = cv2.getTrackbarPos('G', 'image')
            b = cv2.getTrackbarPos('B', 'image')
            s = cv2.getTrackbarPos(switch_trackbar_name, 'image')

            if s == 0:
                # assign zeros to all pixels
                img[:] = 0
            else:
                # assign the same BGR color to all pixels
                img[:] = [b, g, r]

        # closes all windows (usually optional as the script ends anyway)
        cv2.destroyAllWindows()

    def progowanie():
        slodek_img = load_image('slodek.jpg')
        slodek_img_grey = cv2.cvtColor(slodek_img, cv2.COLOR_BGR2GRAY)
        leveled = slodek_img_grey
        cv2.imshow("Radoslaw Slodkiewicz", slodek_img_grey)
        cv2.createTrackbar('level', 'Radoslaw Slodkiewicz', 0, 255, empty_callback)
        cv2.createTrackbar('type', 'Radoslaw Slodkiewicz', 0, 4, empty_callback)
        while True:
            type_p = cv2.getTrackbarPos('type', 'Radoslaw Slodkiewicz')
            level = cv2.getTrackbarPos('level', 'Radoslaw Slodkiewicz')
            match type_p:
                case 0:
                    ret, leveled = cv2.threshold(slodek_img_grey, level, 255, cv2.THRESH_BINARY)
                case 1:
                    ret, leveled = cv2.threshold(slodek_img_grey, level, 255, cv2.THRESH_BINARY_INV)
                case 2:
                    ret, leveled = cv2.threshold(slodek_img_grey, level, 255, cv2.THRESH_TRUNC)
                case 3:
                    ret, leveled = cv2.threshold(slodek_img_grey, level, 255, cv2.THRESH_TOZERO)
                case 4:
                    ret, leveled = cv2.threshold(slodek_img_grey, level, 255, cv2.THRESH_TOZERO_INV)
            cv2.imshow("Radoslaw Slodkiewicz ulepszony", leveled)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    def skalowanie(s=2.75):
        qr_img = load_image('qr.jpg')
        cv2.imshow('Original', qr_img)
        qr_scaled = [("LINEAR", cv2.resize(qr_img, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)),
                     ("NEAREST", cv2.resize(qr_img, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_NEAREST)),
                     ("AREA", cv2.resize(qr_img, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)),
                     ("LANCZOS4", cv2.resize(qr_img, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4))]
        for q in qr_scaled:
            cv2.imshow("INTER_" + q[0], q[1])
        while True:
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    def blending():
        imgs = [load_image('pobranyork.jpeg'),
                load_image('LOGO_PUT_VISION_LAB_MAIN.png')]
        for img in imgs:
            if img is None:
                sys.exit("Could not read the image/s")
        imgs[1] = cv2.resize(imgs[1], dsize=(0, 0), fx=2, fy=2)
        imgs[0] = cv2.resize(imgs[0], (imgs[1].shape[1], imgs[1].shape[0]))
        cv2.imshow('blended', imgs[0])
        cv2.createTrackbar('beta/alpha', 'blended', 0, 1000, empty_callback)
        while True:
            value = cv2.getTrackbarPos('beta/alpha', 'blended')
            img = cv2.addWeighted(imgs[0], 1 - value / 1000, imgs[1], value / 1000, 0)
            cv2.imshow('blended', img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    def scale_time(s=2.75):
        qr_img = load_image('qr.jpg')
        interpolation = [('INTER_LINEAR', cv2.INTER_LINEAR),
                         ('INTER_NEAREST', cv2.INTER_NEAREST),
                         ('INTER_AREA', cv2.INTER_AREA),
                         ('INTER_LANCZOS4', cv2.INTER_LANCZOS4)]
        qr_scaled = []
        for i in interpolation:
            t_start = time.perf_counter()
            qr_scaled.append(cv2.resize(qr_img, dsize=(0, 0), fx=s, fy=s, interpolation=i[1]))
            t_stop = time.perf_counter()
            print(f'Czas skalowania {i[0]} wynosi {t_stop - t_start} ms')

    def negative_fun(img):
        if img is None:
            sys.exit("Could not read the image slodek.jpg")
        cv2.imshow('img', img)
        img_inv = cv2.bitwise_not(img)
        cv2.imshow('img inverted', img_inv)
        while True:
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
    # wywolania

    # trackbar_test()
    # progowanie()
    # skalowanie()
    # blending()
    # scale_time()
    negative_fun(load_image('slodek.jpg'))

# LAB03
def lab03():
    def filterring():
        filenames = ['lenna_noise.bmp', 'lenna_salt_and_pepper.bmp']
        lenna_noise = []
        i = 0
        for filename in filenames:
            lenna_noise.append((load_image(filename), filename, i))
            i = i + 1
        for img in lenna_noise:
            if img[0] is None:
                sys.exit("Could not read the image" + img[1])
            else:
                cv2.imshow(img[1], img[0])
                cv2.createTrackbar('window size ' + img[1], img[1], 1, 5, empty_callback)
        filter_type = ['2D convulsion', 'Gaussian blurring', 'Median blurring']
        size = list([])
        mask = list([])
        done = list([])
        while True:
            if len(size) == 0:
                for img in lenna_noise:
                    size.append(2 * cv2.getTrackbarPos('window size ' + img[1], img[1]) - 1)
                    mask.append(np.ones((size[img[2]], size[img[2]]), np.float32) / size[img[2]] ** 2)
            else:
                for img in lenna_noise:
                    size[img[2]] = 2 * cv2.getTrackbarPos('window size ' + img[1], img[1]) + 1
                    mask[img[2]] = np.ones((size[img[2]], size[img[2]]), np.float32) / size[img[2]] ** 2
            done.clear()
            for typee in filter_type:
                match typee:
                    case '2D convulsion':
                        for img in lenna_noise:
                            done.append((cv2.filter2D(img[0], -1, mask[img[2]]), typee + ' of ' + img[1]))
                    case 'Gaussian blurring':
                        for img in lenna_noise:
                            done.append(
                                (cv2.GaussianBlur(img[0], (size[img[2]], size[img[2]]), 0), typee + ' of ' + img[1]))
                    case 'Median blurring':
                        for img in lenna_noise:
                            done.append((cv2.medianBlur(img[0], size[img[2]]), typee + ' of ' + img[1]))
            for ork in done:
                cv2.imshow(ork[1], ork[0])
            if cv2.waitKey(1) & 0xFF == 27:
                break

    def progowanie():
        slodek_img = load_image('slodek.jpg')
        slodek_img_grey = cv2.cvtColor(slodek_img, cv2.COLOR_BGR2GRAY)
        leveled = slodek_img_grey
        cv2.imshow("Radoslaw Slodkiewicz", slodek_img_grey)
        cv2.createTrackbar('level', 'Radoslaw Slodkiewicz', 0, 255, empty_callback)
        cv2.createTrackbar('type', 'Radoslaw Slodkiewicz', 0, 4, empty_callback)
        cv2.namedWindow('Radoslaw Slodkiewicz ulepszony')
        cv2.createTrackbar('size', 'Radoslaw Slodkiewicz ulepszony', 0, 10, empty_callback)
        while True:
            type_p = cv2.getTrackbarPos('type', 'Radoslaw Slodkiewicz')
            level = cv2.getTrackbarPos('level', 'Radoslaw Slodkiewicz')
            match type_p:
                case 0:
                    ret, leveled = cv2.threshold(slodek_img_grey, level, 255, cv2.THRESH_BINARY)
                case 1:
                    ret, leveled = cv2.threshold(slodek_img_grey, level, 255, cv2.THRESH_BINARY_INV)
                case 2:
                    ret, leveled = cv2.threshold(slodek_img_grey, level, 255, cv2.THRESH_TRUNC)
                case 3:
                    ret, leveled = cv2.threshold(slodek_img_grey, level, 255, cv2.THRESH_TOZERO)
                case 4:
                    ret, leveled = cv2.threshold(slodek_img_grey, level, 255, cv2.THRESH_TOZERO_INV)
            cv2.imshow("Radoslaw Slodkiewicz ulepszony", leveled)
            window_size = cv2.getTrackbarPos('size', 'Radoslaw Slodkiewicz ulepszony') + 1
            kernel = np.ones((window_size, window_size), np.uint8)
            slodek_erosion = cv2.erode(leveled, kernel, iterations=1)
            slodek_dilation = cv2.dilate(leveled, kernel, iterations=1)
            slodek_opening = cv2.morphologyEx(leveled, cv2.MORPH_OPEN, kernel)
            slodek_closing = cv2.morphologyEx(leveled, cv2.MORPH_CLOSE, kernel)
            cv2.imshow("Radoslaw Slodkiewicz ulepszony - erosion", slodek_erosion)
            cv2.imshow("Radoslaw Slodkiewicz ulepszony - dilation", slodek_dilation)
            cv2.imshow("Radoslaw Slodkiewicz ulepszony - opening", slodek_opening)
            cv2.imshow("Radoslaw Slodkiewicz ulepszony - closing", slodek_closing)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    def skanowanie():
        slodek_img = load_image('slodek.jpg')
        slodek_img_grey = cv2.cvtColor(slodek_img, cv2.COLOR_BGR2GRAY)
        slodek_img_grey_fix = cv2.cvtColor(slodek_img, cv2.COLOR_BGR2GRAY)
        for row in range(slodek_img_grey.shape[0]):
            for column in range(slodek_img_grey.shape[1]):
                if column % 3 == 0:
                    slodek_img_grey[row][column] = 255
        value = list([])
        kernel = np.ones((3, 3), np.float32) / 9
        t_start = time.perf_counter()
        for row in range(slodek_img_grey.shape[0] - 2):
            for column in range(slodek_img_grey.shape[1] - 2):
                for x in range(row, row + 3):
                    for y in range(column, column + 3):
                        value.append(slodek_img_grey[x][y])
                slodek_img_grey_fix[row + 1][column + 1] = round(sum(value) / 9)
                value.clear()
        t_stop = time.perf_counter()
        print(f'Czas wygladzania wlasnego wynosi {t_stop - t_start} ms')
        t_start = time.perf_counter()
        slodek_img_grey_blur_fix = cv2.blur(slodek_img_grey, (3, 3))
        t_stop = time.perf_counter()
        print(f'Czas wygladzania wbudowanego wynosi {t_stop - t_start} ms')
        t_start = time.perf_counter()
        slodek_img_grey_2D_fix = cv2.filter2D(slodek_img_grey, -1, kernel)
        t_stop = time.perf_counter()
        print(f'Czas wygladzania wbudowanego 2D wynosi {t_stop - t_start} ms')
        while True:
            cv2.imshow('Radoslaw Slodkiewicz zepsuty', slodek_img_grey)
            cv2.imshow('Radoslaw Slodkiewicz naprawa moja', slodek_img_grey_fix)
            cv2.imshow('Radoslaw Slodkiewicz naprawa blur', slodek_img_grey_blur_fix)
            cv2.imshow('Radoslaw Slodkiewicz naprawa 2D', slodek_img_grey_2D_fix)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # wywolania

    # filterring()
    # progowanie()
    skanowanie()

# LAB04
def lab04():
    def rysowanie():
        def policja_rysuje(event, x, y, flags, param):
            width = 50
            height = 50
            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.rectangle(slodek_img, ((x + width), (y + height)), ((x - width), (y - height)), (255, 0, 0), 4)
        slodek_img = load_image('slodek.jpg')
        cv2.namedWindow('slodek')
        cv2.setMouseCallback('slodek', policja_rysuje)

        while True:
            cv2.imshow('slodek', slodek_img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    def transformacje_geometryczne(scale=0.6):
        road_img = load_image('road.jpg')
        road_img_scaled = cv2.resize(road_img, (0, 0), fx=scale, fy=scale)
        pts = list([])

        def narozoniki_skala():
            def mousecallback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    pts.append((x, y))

            def order_points(pts):
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                pts = np.delete(pts, np.argmin(s), 0)
                rect[2] = pts[np.argmax(s)]
                pts = np.delete(pts, np.argmax(s), 0)
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmax(diff)]
                rect[3] = pts[np.argmin(diff)]
                return rect

            def convert_points(pts):
                rect = np.zeros((4, 2), dtype="float32")
                rect[0] = pts[0]
                rect[2] = pts[2]
                rect[1] = pts[1]
                rect[3] = pts[3]
                return rect

            cv2.setMouseCallback('Road', mousecallback)
            if len(pts) == 4:
                points = np.array(pts)
                rect = convert_points(points)
                (tl, tr, br, bl) = rect

                width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                width_max = max(int(width1), int(width2))

                height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                height_max = max(int(height1), int(height2))

                dst = np.array([
                    [0, 0],
                    [width_max - 1, 0],
                    [width_max - 1, height_max - 1],
                    [0, height_max - 1]], dtype="float32")
                # compute the perspective transform matrix and then apply it
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(road_img_scaled, M, (width_max, height_max))

                cv2.imshow('u', warped)

        while True:
            cv2.imshow('Road', road_img_scaled)
            narozoniki_skala()

            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    def histogram(color='colorscale'):
        global slodek_img_equ
        path = 'C:/Users/molsz/OneDrive/PUT/WDPO/data/img/slodek.jpg'
        match color:
            case 'greyscale':
                slodek_img = load_image('slodek.jpg', cv2.IMREAD_GRAYSCALE)
                hist = cv2.calcHist([slodek_img], [0], None, [256], [0, 256])
                plt.plot(hist)
                plt.title('histogram Radoslaw Slodkiewicz')
                plt.show()
                slodek_img_equ = cv2.equalizeHist(slodek_img)
                res = np.hstack((slodek_img, slodek_img_equ))
                hist = cv2.calcHist([slodek_img_equ], [0], None, [256], [0, 256])
                plt.plot(hist, color='r')
                plt.title('histogram Radoslaw Slodkiewicz ulepszony')
                plt.show()
            case 'colorscale':
                slodek_img = load_image('slodek.jpg')
                color = ('b', 'g', 'r')
                for i, col in enumerate(color):
                    histr = cv2.calcHist([slodek_img], [i], None, [256], [0, 256])
                    plt.plot(histr, color=col)
                    plt.xlim([0, 256])
                    plt.show()
            case _:
                sys.exit('argument can only take: greyscale/colorscale')

        while True:
            cv2.imshow('Radoslaw Slodkiewicz', slodek_img)
            if color == 'greyscale':
                cv2.imshow('Radoslaw Slodkiewicz wyrownany', slodek_img_equ)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    # zadania do samodzielnej realizacji
    def zad1():
        ork_img = load_image('pobranyork.jpeg', cv2.IMREAD_GRAYSCALE)

    # wywolania
    # rysowanie()
    # transformacje_geometryczne()
    # histogram('greyscale')

# LAB05
def lab05():
    # some needed shit
    def img_zad_1_2(filename='logo.png', scale=1.0):
        img = load_image(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        kernel_x_pre = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]])
        kernel_y_pre = np.array([[1, 1, 1],
                                 [0, 0, 0],
                                 [-1, -1, -1]])
        kernel_x_sob = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        kernel_y_sob = np.array([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]])
        img_prewittx = cv2.filter2D(img, -1, kernel_x_pre)
        img_prewitty = cv2.filter2D(img, -1, kernel_y_pre)
        img_sobelx = cv2.filter2D(img, -1, kernel_x_sob)
        img_sobely = cv2.filter2D(img, -1, kernel_y_sob)
        return img, img_prewittx, img_prewitty, img_sobelx, img_sobely
    # rozwiazania zadan
    def zad01():
        edd_img, img_prewittx, img_prewitty, img_sobelx, img_sobely = img_zad_1_2(filename='edd.png', scale=0.4)
        while True:
            cv2.imshow('oryginal', edd_img)
            cv2.imshow('prewitt_x', img_prewittx)
            cv2.imshow('prewitt_y', img_prewitty)
            cv2.imshow('sobel_x', img_sobelx)
            cv2.imshow('sobel_y', img_sobely)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    def zad02():
        edd_img, img_prewittx, img_prewitty, img_sobelx, img_sobely = img_zad_1_2(filename='edd.png', scale=0.4)
        m_pre = np.sqrt(img_prewitty ** 2 + img_prewittx ** 2)
        k_pre = np.amax(m_pre)
        m_pre = (m_pre) / k_pre * 255
        m_pre = m_pre.astype(dtype=np.uint8)
        m_sob = np.sqrt(img_sobely ** 2 + img_sobelx ** 2)
        k_sob = np.amax(m_sob)
        m_sob = m_sob / k_sob * 255
        m_sob = m_sob.astype(dtype=np.uint8)
        while True:
            cv2.imshow('pre_grad', m_pre)
            cv2.imshow('sob_grad', m_sob)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    def zad3():
        img = load_image(filename='slodek.jpg', scale=0.8)
        cv2.namedWindow('original')
        cv2.createTrackbar('threshold1', 'original', 0, 255, empty_callback)
        cv2.createTrackbar('threshold2', 'original', 0, 255, empty_callback)
        while True:
            cv2.imshow('original', img)
            thresh1 = cv2.getTrackbarPos('threshold1', 'original')
            thresh2 = cv2.getTrackbarPos('threshold2', 'original')
            edges = cv2.Canny(img, threshold1=thresh1, threshold2=thresh2)
            cv2.imshow('edges', edges)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    def zad4():
        shapes_img = load_image('shapes.jpg', scale=0.75)
        shapes_img_c = cv2.cvtColor(shapes_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(shapes_img, 100, 200)
        result_lines = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result_circles = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=35, minLineLength=30, maxLineGap=3)
        circles = cv2.HoughCircles(shapes_img_c, cv2.HOUGH_GRADIENT, dp=1, minDist=60, param1=50, param2=50, minRadius=0, maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(result_circles, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(result_circles, (i[0], i[1]), 2, (0, 0, 255), 3)
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(result_lines, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv2.LINE_AA)
        while True:
            cv2.imshow('edges', edges)
            cv2.imshow('result_linesP', result_lines)
            cv2.imshow('resut_circles', result_circles)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    def zad5():
        ship_img = load_image('drone_ship.jpg', scale=1)
        edges = cv2.Canny(ship_img, threshold1=200, threshold2=255)
        lines = cv2.HoughLinesP(image=edges, rho=3, theta=np.pi / 360, threshold=100, minLineLength=30, maxLineGap=5)
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(ship_img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
        while True:
            #cv2.imshow('edges', edges)
            cv2.imshow('ship', ship_img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # wywolania
    #zad01()
    #zad02()
    #zad3()
    #zad4()
    zad5()

if __name__ == '__main__':
    # lab01()
    # lab02()
    # lab03()
    # lab04()
    lab05()
