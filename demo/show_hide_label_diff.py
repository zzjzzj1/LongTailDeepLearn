import json
import os
from sklearn.externals import joblib
from random import sample

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


def split_count(data_set):
    """
    data_set look like
    {
        1: [CountResult, CountResult],
        2: [CountResult, CountResult, CountResult, ...., CountResult]
        .
        .
        .
        29: [CountResult, CountResult, CountResult, ...., CountResult]
    }
    """
    image_max = 0
    type_group_data = []
    for key in data_set:
        image_max += len(data_set[key])
        type_group_data.append([key, len(data_set[key])])
    type_group_data.sort(key=lambda x: x[1], reverse=True)
    long_tail_mock = get_img_num_per_cls(type_group_data[0][1] / 9, 1.0 / 100, len(data_set))
    for i in range(len(data_set)):
        # judge real dateset have more data than want mock?
        if long_tail_mock[i] > type_group_data[i][1]:
            print('error!')
    count_hide_label = {}
    for i in range(global_var.coco_contributes_hide_type_number):
        count_hide_label[i] = {}
    for i in range(len(long_tail_mock)):
        key = type_group_data[i][0]
        # random choice data and count hide label
        data_choice = sample(data_set[key], long_tail_mock[i])
        for item in data_choice:
            hide_label = item.hide_label
            for j in range(len(hide_label)):
                wait_count_dict = count_hide_label[j]
                if hide_label[j] not in wait_count_dict:
                    wait_count_dict[hide_label[j]] = 0
                wait_count_dict[hide_label[j]] += 1
    count_result = {}
    for key in count_hide_label:
        temp = [count_hide_label[key][hide_label_id] for hide_label_id in count_hide_label[key]]
        temp.sort(reverse=True)
        count_result[key] = temp
    for key in count_result:
        print(count_result[key])


def count(contributes, coco_old_data):
    coco_attr_vecs = contributes['ann_vecs']
    coco_annotations_dict = {}
    # get map relation
    for coco_entity in coco_old_data['annotations']:
        coco_annotations_dict[coco_entity['id']] = coco_entity
    data_set = {}
    for key in coco_attr_vecs:
        coco_id = contributes['patch_id_to_ann_id'][key]
        # judge coco_dataset hava enhance attributes
        if coco_id not in coco_annotations_dict:
            continue
        item = solve_map(coco_annotations_dict[coco_id], coco_attr_vecs[key])
        if item.main_label not in data_set:
            data_set[item.main_label] = []
        data_set[item.main_label].append(item)
    split_count(data_set)


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
