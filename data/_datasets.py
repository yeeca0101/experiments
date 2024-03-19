import os
from typing import Callable, Any,List
import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO

#############################################################################################################

class CIFAR10(datasets.CIFAR10):
    def __init__(self, train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None) -> None:
        curr_dir=  get_current_script_directory()
        root=os.path.join(curr_dir,'dataset/CIFAR10')
        download=False

        super().__init__(root, train, transform, target_transform, download)

class MNIST(datasets.MNIST):
    def __init__(self, train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None) -> None:
        curr_dir=  get_current_script_directory()
        root=os.path.join(curr_dir,'dataset/MNIST')
        download=False

        super().__init__(root, train, transform, target_transform, download)

###################################################################################################
        

_type = {'train':'train','test':'test','valid':'val'}

class COCOBase(Dataset):
    def __init__(self,
                 root:str|None= None,
                 anno_file:str|None= None,
                 split_type:str|None= None,
                 anno_type:str|None= None,
                 transform: Callable[..., Any]| None = None,
                 target_transform: Callable[..., Any] | None = None
                 ) -> None:
        super().__init__()

        # images path
        self.root = root if (root is not None) or '' else 'dataset/COCO/images'
        self.root = os.path.join(get_current_script_directory(),self.root,_type[split_type]+'2017')

        # annotations path
        self.anno_path = 'dataset/COCO/annotations'
        if anno_file is not None:
            self.anno_path = os.path.join(get_current_script_directory(),self.anno_path,anno_file)
        else:
            self._set_annotation_path(anno_type,split_type)

        # get coco api
        self.coco = COCO(annotation_file=self.anno_path)

    def _set_annotation_path(self,anno_type,split_type):
        # TODO: stuff 
        if anno_type not in {'caption', 'instance', 'keypoint'}:
                raise ValueError("Invalid value for 'anno_type'. Should be one of 'caption', 'instance', or 'keypoint'.")
        else:
            if split_type not in {'train','test','valid'}:
                raise ValueError("Invalid value for 'split_type'. Should be one of 'train', 'valid', or 'test'.")
            else:
                if anno_type == 'caption':
                    _format = f'captions_{_type[split_type]}2017.json'
                elif anno_type == 'instance':
                    _format = f'instances_{_type[split_type]}2017.json'
                elif anno_type == 'keypoint':
                    _format = f'person_keypoints_{_type[split_type]}2017.json'
                if split_type == 'test':
                    _format = 'image_info_test2017.json'
                    
        self.anno_path = os.path.join(get_current_script_directory(),self.anno_path,_format)

    

class COCOSegDataset(COCOBase):
    """
    ref : https://github.com/virafpatrawala/COCO-Semantic-Segmentation/blob/master/COCOdataset_SemanticSegmentation_Demo.ipynb
    """
    def __init__(self,
                 split_type:str|None= None,
                 anno_type:str|None= None,
                 filter_classes:List[str]|None= None,
                 transform: Callable[..., Any]| None = None,
                 target_transform: Callable[..., Any] | None = None
                 ) -> None:
        super().__init__(split_type=split_type,
                         anno_type=anno_type,
                         transform=transform,
                         target_transform=target_transform)
        
        # set used classes : max (80)
        # setting self.classes, self.class_ids, self.categori_dict
        self._set_used_classes(filter_classes,split_type)

        # load imgs
        img_ids = self.coco.getImgIds(catIds=self.class_ids)
        self.ids = list(sorted(img_ids))
        self.categori_dict = self.coco.loadCats(self.class_ids)

        # id to class
        self.id2class_dict = {item['id']: item['name'] for item in self.categori_dict}

    def _is_valid_categori_ids(self,categori_ids):
        return categori_ids != []

    def _chk_valid_filter_classes(self,overal_classes,filter_classes):
        not_exist_elements = [a for a in filter_classes if a not in overal_classes]
        if len(not_exist_elements) > 0:
            raise ValueError(f'The following elements are not in the dataset: {not_exist_elements}')

    def _set_used_classes(self,filter_classes:List|None=None,split_type='train'):
            '''for instance or stuff types'''
            # overal classes
            categori_ids = self.coco.getCatIds()
            self.categori_dict = self.coco.loadCats(categori_ids)
            class_list = [name['name'] for name in self.categori_dict]
            # specific classes
            if filter_classes is not None and split_type != 'test':
                class_list = [f_element for f_element in filter_classes if f_element in class_list]
            
            self._chk_valid_filter_classes(class_list,filter_classes)

            self.filter_classes = filter_classes
            self.classes = class_list
            self.class_ids = self.coco.getCatIds(catNms=class_list)
            self.categori_dict = self.coco.loadCats(self.class_ids)

    def _get_info_seg(self,idx):
        coco = self.coco
        # image id <scalar>
        img_id = self.ids[idx]
        # annotations id >= 1 <list>
        # annotation of target categories
        ann_ids = coco.getAnnIds(imgIds=img_id,catIds=self.id2class_dict.keys())
        if len(ann_ids) < 1:
            print('not exist') 
        print(ann_ids)
        # load annotations 
        annotations = coco.loadAnns(ann_ids) 
        # return
        # load image
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        img_h = img_info['height']
        img_w = img_info['width']
        img = Image.open(os.path.join(self.root,img_filename))

        # extract seg info
        cls_names = []
        num_objs = len(annotations)
        mask = np.zeros((img_h,img_w))
        for i in range(num_objs):
            try:
                className = self.id2class_dict[annotations[i]['category_id']]
            except:
                print(self.id2class_dict)
                print(annotations[i]['category_id'])

            cls_names.append(className)
            pixel_value = self.filter_classes.index(className)+1
            mask = np.maximum(coco.annToMask(annotations[i])*pixel_value, mask)

        import matplotlib.pyplot as plt
        print('Unique pixel values in the mask are:', np.unique(mask))
        print(cls_names)
        plt.imshow(mask)
        plt.show()

        return mask
    
# utils
def get_current_script_directory():
    # /root/data
    return os.path.dirname(os.path.realpath(__file__))

# test
if __name__ == '__main__':

    def mnist_test():
        c = MNIST(train=True)

    def cocobase_test():
        c = COCOBase(anno_type='instance',split_type='train')
        print(len(c.ids)) # 118287

        c = COCOBase(anno_type='keypoint',split_type='train')
        print(len(c.ids)) # 118287

        c = COCOBase(anno_type='caption',split_type='train')
        print(len(c.ids)) # 118287

        # train : 118,287 / valid : 5,000
    
    def cocobase_filter_class_test():
        c = COCOBase(anno_type='instance',split_type='valid',filter_classes=['laptop', 'person'])
        print(len(c.ids)) # 118287
        print(c.id2class_dict)

    def cocobase_filter_class_mask_test():
        c = COCOBase(anno_type='instance',split_type='valid',filter_classes=None)
        print(len(c.ids)) # 118287
        print(c.id2class_dict)

    cocobase_filter_class_mask_test()