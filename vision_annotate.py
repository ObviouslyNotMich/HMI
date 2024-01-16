import cv2
import os

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img2 = img.copy()
            cv2.rectangle(img2, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', img2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        save_annotation(ix, iy, x, y)


def save_annotation(x1, y1, x2, y2):
    global img_file
    with open('positives.txt', 'a') as f:
        f.write(f'{img_file} 1 {x1} {y1} {x2-x1} {y2-y1}\n')


base_folder = 'F:/Code/HMIvoice/image/p'
drawing = False
ix, iy = -1, -1

for dirpath, dirnames, filenames in os.walk(base_folder):
    for filename in filenames:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_file = os.path.join(dirpath, filename)
            img = cv2.imread(img_file)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('image', draw_rectangle)

            while True:
                cv2.imshow('image', img)
                key = cv2.waitKey(20)
                if key & 0xFF == 27: # Escape key to close the window
                    break

            cv2.destroyAllWindows()
