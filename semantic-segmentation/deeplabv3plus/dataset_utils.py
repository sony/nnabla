import numpy as np
import os
from scipy.misc import imread
from args import get_args
import matplotlib.pyplot as plt

def get_color():
    #RGB format
    return np.array([[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],[120,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],[64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],[192,128,128],[0,64,0],[128,64,0],[0,192,0],[128,192,0],[0,64,128],[224,224,192]])



def encode_label(label):

    '''
    Converting pixel values to corresponding class numbers. Assuming that the input label in 3-dim(h,w,c) and in BGR fromat read from cv2
    '''

    h,w,c = label.shape
    new_label = np.zeros((h,w,1), dtype=np.int32)

    cls_to_clr_map = get_color()

    for i in range(cls_to_clr_map.shape[0]):
        #new_label[(label == cls_to_clr_map[i])[:,:,0]] = i
        #new_label[np.argwhere((label.astype(np.int32) == cls_to_clr_map[i]).all(axis=2))]=i
        print(np.where((label.astype(np.int32) == [120,0,128]).all(axis=2)))
        if i==21:
            new_label[np.where((label.astype(np.int32) == cls_to_clr_map[i]).all(axis=2))]=255
        else:
            new_label[np.where((label.astype(np.int32) == cls_to_clr_map[i]).all(axis=2))]=i

    return new_label



#this method should generate train-image.txt and train-label.txt 
def generate_path_files(data_dir, train_file, val_file):

    ti = open('train_image.txt', 'w')
    tl = open('train_label.txt', 'w')
    vi = open('val_image.txt', 'w')
    vl = open('val_label.txt', 'w')

    rootdir = data_dir

    train_text_file = open(train_file, "r")
    lines = [line[:-1] for line in train_text_file]
    for line in lines:
        if os.path.exists(data_dir+'JPEGImages/'+line+'.jpg'):
            ti.write(data_dir+'JPEGImages/'+line+'.jpg' + '\n')
            assert (os.path.isfile(data_dir+'SegmentationClass/encoded/'+line + '.npy')), "No matching label file for image : " + line + '.jpg'
            tl.write(data_dir+'SegmentationClass/encoded/'+line + '.npy' + '\n')

    val_text_file = open(val_file, "r")
    lines = [line[:-1] for line in val_text_file]
    for line in lines:
        if os.path.exists(data_dir+'JPEGImages/'+line+'.jpg'):
            vi.write(data_dir+'JPEGImages/'+line+'.jpg' + '\n')
            assert (os.path.isfile(data_dir+'SegmentationClass/encoded/'+line + '.npy')), "No matching label file for image : " + line + '.jpg'
            vl.write(data_dir+'SegmentationClass/encoded/'+line + '.npy' + '\n')

    ti.close()
    tl.close()
    vi.close()
    vl.close()


def main():
    '''
    Arguments:
    train-file = txt file containing randomly selected image filenames to be taken as training set.
    val-file = txt file containing randomly selected image filenames to be taken as validation set.
    data-dir = dataset directory
    Usage: python dataset_utils.py --train-file="" --val-file="" --data_dir=""
    '''

    args = get_args()
    data_dir=args.data_dir
     
    if not os.path.exists(data_dir+'SegmentationClass/' + 'encoded/'):
        os.makedirs(data_dir+'SegmentationClass/' + 'encoded/')
    for filename in os.listdir(data_dir+'SegmentationClass/'):
        if os.path.isdir(data_dir+'SegmentationClass/' + filename):
            continue
        label = imread(data_dir+'SegmentationClass/' + filename).astype('float32')
        label = encode_label(label)
        np.save(data_dir+'SegmentationClass/' + 'encoded/' + filename.split('.')[0] + '.npy',label)

    
    generate_path_files(args.data_dir, args.train_file, args.val_file)

if __name__ == '__main__':
    main()
