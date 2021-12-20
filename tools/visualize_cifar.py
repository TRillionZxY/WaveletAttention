import os
import pickle
#import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
 
def load_labels_name(filename):
    """使用pickle反序列化labels文件，得到存储内容
        cifar10的label文件为“batches.meta”，cifar100则为“meta”
        反序列化之后得到字典对象，可根据key取出相应内容
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj
        
def load_data_cifar(filename, mode='cifar10'):
    """ load data and labels information from cifar10 and cifar100
    cifar10 keys(): dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    cifar100 keys(): dict_keys([b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data'])
    """
    with open(filename,'rb') as f:
        dataset = pickle.load(f, encoding='bytes')
        if mode == 'cifar10':
            data = dataset[b'data']
            labels = dataset[b'labels']
            img_names = dataset[b'filenames']
        elif mode == 'cifar100':
            data = dataset[b'data']
            labels = dataset[b'fine_labels']
            img_names = dataset[b'filenames']
        else:
            print("mode should be in ['cifar10', 'cifar100']")
            return None, None, None
        
    return data, labels, img_names
 
def load_cifar10(cifar10_path, mode = 'train'):
    
    if mode == "train":
        data_all = np.empty(shape=[0, 3072],dtype=np.uint8)
        labels_all = []
        img_names_all = []
        for i in range(1,6):
            filename = os.path.join(cifar10_path, 'data_batch_'+str(i)).replace('\\','/')
            print("Loading {}".format(filename))
            data, labels, img_names = load_data_cifar(filename, mode='cifar10')
            data_all = np.vstack((data_all, data))
            labels_all += labels
            img_names_all += img_names
        return data_all,labels_all,img_names_all
    elif mode == "test":
        filename = os.path.join(cifar10_path, 'test_batch').replace('\\','/')
        print("Loading {}".format(filename))
        return load_data_cifar(filename, mode='cifar10')
        
 
def load_cifar100(cifar100_path, mode = 'train'):
    if mode == "train":
        filename = os.path.join(cifar100_path, 'train')
        print("Loading {}".format(filename))
        data, labels, img_names = load_data_cifar(filename, mode='cifar100')
    elif mode == "test":
        filename = os.path.join(cifar100_path, 'test')
        print("Loading {}".format(filename))
        data, labels, img_names = load_data_cifar(filename, mode='cifar100')
    else:
        print("mode should be in ['train', 'test']")
        return None, None, None
    
    return data, labels, img_names
    
def to_pil(data):
    r = Image.fromarray(data[0])
    g = Image.fromarray(data[1])
    b = Image.fromarray(data[2])
    pil_img = Image.merge('RGB', (r,g,b))
    return pil_img
 
def random_visualize(imgs, labels, label_names):
    figure = plt.figure(figsize=(len(label_names),10))
    idxs = list(range(len(imgs)))
    np.random.shuffle(idxs)
    count = [0]*len(label_names)
    for idx in idxs:
        label = labels[idx]
        if count[label]>=10:
            continue
        if sum(count)>10 * len(label_names):
            break
        
        img = to_pil(imgs[idx])
        label_name = label_names[label]
        
        subplot_idx = count[label] * len(label_names) + label + 1
        print(label, subplot_idx)
        plt.subplot(10,len(label_names), subplot_idx)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        if count[label] == 0:
            plt.title(label_name)
 
        count[label] += 1
    
    plt.show()
        
    
 
if __name__ == "__main__":
    # 修改为你的数据集存放路径
    #10: .../cifar-10-batches-py
    #100: .../cifar-100-python
    cifar10_path = ""
    cifar100_path = ""
    
    obj_cifar10 = load_labels_name(os.path.join(cifar10_path, 'batches.meta')) # label_names、num_cases_per_batch、num_vis
    #obj_cifar100 = load_labels_name(os.path.join(cifar100_path, 'meta')) # coarse_label_names、fine_label_names
    
 
    # 提取cifar10、cifar100的图片数据、标签、文件名
    # data_cifar10_train,labels_cifar10_train,img_names_cifar10_train = \
    #                             load_cifar10(cifar10_path, mode='train')
    data_cifar10_test,labels_cifar10_test,img_names_cifar10_test = \
                                load_cifar10(cifar10_path, mode='test')
    # imgs_cifar10_train = data_cifar10_train.reshape(data_cifar10_train.shape[0],3,32,32)
    imgs_cifar10_test = data_cifar10_test.reshape(data_cifar10_test.shape[0],3,32,32)
    
    # data_cifar100_train,labels_cifar100_train,img_names_cifar100_train = \
    #                             load_cifar100(cifar100_path, mode = 'train')
    # data_cifar100_test,labels_cifar100_test,img_names_cifar100_test = \
    #                             load_cifar100(cifar100_path, mode = 'test')
    # imgs_cifar100_train = data_cifar100_train.reshape(data_cifar100_train.shape[0],3,32,32)
    # imgs_cifar100_test = data_cifar100_test.reshape(data_cifar100_test.shape[0],3,32,32)
 
    # visualize fro cifar10
    label_names_cifar10 = obj_cifar10['label_names']
    print(label_names_cifar10)
    # random_visualize(imgs=imgs_cifar10_train, 
    #                  labels=labels_cifar10_train, 
    #                  label_names=label_names_cifar10)
    
    # visualize fro cifar100
    # label_names_cifar100 = obj_cifar100['fine_label_names']
    # random_visualize(imgs=imgs_cifar100_train, 
    #                  labels=labels_cifar100_train, 
    #                  label_names=label_names_cifar100)

    with open('img_label.txt', 'a+') as f:
        for i in range(200):
            f.write(str(img_names_cifar10_test[i], encoding="utf-8")+' '+str(labels_cifar10_test[i])+'\n')

    for i in range(200):
        img = to_pil(imgs_cifar10_test[i])
        name = str(img_names_cifar10_test[i], encoding="utf-8")
        img.save("./pil/"+name,"png")
    print("save successfully!")