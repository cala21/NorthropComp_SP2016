#########################################################
# Given a a test image will return classified image
# If given image is part of dataset will also return truth data
#
#########################################################

import sys, getopt, os

DATA_DIR = '../Dataset/FinalRawData/'
LABELED_DIR = '../Dataset/FinalLabeledData/'

def classify_image(img):
    truth = check_dataset(img)
    classified = img

    return classified, truth

def check_dataset(img):
    abs_path = os.path.dirname(os.path.abspath(__file__))
    path_to_labeled = os.path.join(abs_path, LABELED_DIR)
    path_to_raw = os.path.join(abs_path, DATA_DIR)
    
    truth_images = sorted(os.listdir(path_to_labeled))
    raw_images = sorted(os.listdir(path_to_raw))

    for raw, labeled in zip(raw_images,truth_images):
        check_img = open_image(path_to_raw+raw)
        if check_img == img:
            truth_img = open_image(path_to_labeled+labeled)
            print("Found truth data for given image")
            return truth_img

    return

def open_image(path):
    try:
        return open(path, "rb").read()
    except:
        print("Unable to open image: \n{0} \nExiting!".format(path))
        exit(0)

def handle_arguments():
    if len(sys.argv) < 3:
        print_usage()

    myopts, args = getopt.getopt(sys.argv[1:],"i:h:")
    img_path=''

    for o, a in myopts:
        if o == '-i':
            img_path=a
        elif o == '-h':
            print_usage()
        else:
            print_usage()

    print("\nImage path: {0}\n".format(img_path))
    img = open_image(img_path) 

    return img
            
def print_usage():
    print("Usage: {0} -i <input image>".format(sys.argv[0]))
    exit(0)


def main():
    img = handle_arguments()

    classified, truth = classify_image(img)

    return 0

if __name__ == '__main__':
    main()
