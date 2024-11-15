from PIL import Image
from frcnn import FRCNN

def picture(image):
    frcnn = FRCNN()
    count = True
    crop = True
    count = True
    r_image,all_labels, all_confs, all_boxes = frcnn.detect_image(image, crop = crop, count = count)
    #r_image.show()
    print(all_labels, all_confs, all_boxes)
    return r_image,all_labels, all_confs, all_boxes

# if __name__ == '__main__':
#     image = picture(r"D:\Car_detection\FastRCNN\img\vid_5_420.jpg")
    
    
