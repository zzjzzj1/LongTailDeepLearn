import json
import os
from sklearn.externals import joblib

import global_var
from global_var import coco_dataset_dir, coco_data_types, coco_contributes_jbl_path


class CountResult:

    def __init__(self, main_label, hide_label):
        # split five types then example CountResult is {main_label: 10, hide_label: [2, 33, 77, 10, 2]}
        self.main_label = main_label
        self.hide_label = hide_label
        pass


def get_img_num_per_cls(img_max, imb_factor, cls_num=100):
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


def solve_map(coco_entity, coco_contributes_entity):
    category_id = get_category_id(coco_entity)
    hide_label = get_hide_label(coco_contributes_entity)
    return CountResult(category_id, hide_label)


def count(contributes, coco_old_data):
    coco_attr_vecs = contributes['ann_vecs']
    coco_annotations_dict = {}
    # get map relation
    for coco_entity in coco_old_data['annotations']:
        coco_annotations_dict[coco_entity['id']] = coco_entity
    ans = {}
    for key in coco_attr_vecs:
        coco_id = contributes['patch_id_to_ann_id'][key]
        # judge coco_dataset hava enhance attributes
        if coco_id not in coco_annotations_dict:
            continue
        item = solve_map(coco_annotations_dict[coco_id], coco_attr_vecs[key])


def get_category_id(coco_entity):
    return coco_entity['category_id']


def get_hide_label(coco_contributes_entity):
    contributes_attr_type = global_var.coco_contributes_attr_type
    record = {}
    for i in range(global_var.coco_contributes_hide_type_number):
        record[i] = [None, -1]
    for index, item in enumerate(coco_contributes_entity):
        temp = record[contributes_attr_type[index]]
        if temp[1] < item:
            temp[0] = index
            temp[1] = item
    return [record[i][0] for i in range(global_var.coco_contributes_hide_type_number)]


if __name__ == '__main__':
    coco_data = {}
    # Change this to location where COCO dataset lives
    for dt in coco_data_types:
        annFile = os.path.join(coco_dataset_dir, 'instances_%s.json' % dt)
        with open(annFile, 'r') as f:
            tmp = json.load(f)
            if coco_data == {}:
                coco_data = tmp
            else:
                coco_data['images'] += tmp['images']
                coco_data['annotations'] += tmp['annotations']

    # Load COCO Attributes
    coco_contributes = joblib.load(coco_contributes_jbl_path)
    # Index of example instance to print
    count(coco_contributes, coco_data)
