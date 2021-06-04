import cv2
def click_event(event, x, y, flags, params):
  
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.imshow('image', img)
  
if __name__=="__main__":
    img = cv2.imread('visualise\\images\\coco2.png', 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()