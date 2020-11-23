import os
import json


#get all of the path data here
coco_path = '/playpen1/home/avmendoz/coco/imgs/'
vg_path  = '/playpen1/home/avmendoz/vg/imgs/'

coco_imgs = (os.path.join(coco_path, "train2017"), os.path.join(coco_path, "val2017"))
vg_data = json.load(open(f'{vg_path[:-5]}/image_data.json'))

vg_only_urls = [entry["url"] for entry in vg_data if entry["coco_id"] is None]
vg_coco_urls= [entry["url"] for entry in vg_data if entry["coco_id"] is not None]

split_stuff = lambda x: os.path.join(vg_path,  '/'.join(x.split("/")[-2:]))
coco_combine = lambda x, y: os.path.join(coco_imgs[y], x)

#these are the three pathes that we want
#expand capability of dataloader to handle list of image paths, or, list of image directories
vg_only_data = set([split_stuff(x) for x in vg_only_urls])
coco_val_data =set([coco_combine(x, 1) for x in os.listdir(coco_imgs[1])])
coco_train_data = set([coco_combine(x, 0) for x in os.listdir(coco_imgs[0])])

#vg_only_ids= set([entry["url"].split('/')[-1][:-4] for entry in vg_data if entry["coco_id"] is None])
#vg_coco_ids = set([entry["url"].split('/')[-1][:-4] for entry in vg_data if entry["coco_id"] is not None])
#coco_get_ids = lambda x: x.split('.')[0]
#vg_coco_data = set([split_stuff(x) for x in vg_coco_urls])
#coco_val_ids = set([coco_get_ids(x) for x in os.listdir(coco_imgs[1])])
#coco_train_ids = set([coco_get_ids(x) for x in os.listdir(coco_imgs[0])])

'''
mval_coco_lxmert = set(map(lambda x: x["img_id"].split('_')[-1], json.load(open('/playpen1/home/avmendoz/lxmertdata/mscoco_minival.json'))))
val_coco_lxmert = set(map(lambda x: x["img_id"].split('_')[-1],json.load(open('/playpen1/home/avmendoz/lxmertdata/mscoco_nominival.json'))))
coco_train_lxmert = set(map(lambda x: x["img_id"].split('_')[-1],
    json.load(open('/playpen1/home/avmendoz/lxmertdata/mscoco_train.json'))
))

# vg_lxmert = set(map(lambda x: x["img_id"],
#     json.load(open('/playpen1/home/avmendoz/lxmertdata/vgnococo.json'))))

# print(len(vg_lxmert.intersection(vg_only_ids)))
#print(len(vg_lxmert))

#check to see if all are present in val data
#will need to check val split + train split

vg_coco_ids = set(map(lambda x: "0"*(12-len(x)) + x, vg_coco_ids))
print(next(iter(vg_coco_ids)))
#print(len(coco_train_lxmert.intersection(coco_val_ids)))
print("num ids not covered in og coco  ids", len(val_coco_lxmert) - len(val_coco_lxmert.intersection(coco_val_ids)))
print("num mval ids not covered in og coco  ids", len(mval_coco_lxmert) - len(mval_coco_lxmert.intersection(coco_val_ids)))
#print("num unsed train ids", len(coco_train_ids) - len(coco_train_lxmert.intersection(coco_train_ids)))
print("num unsed train ids", len(coco_train_ids) - len(coco_train_lxmert.intersection(coco_train_ids)))
print("num unsuded train that are in vg", len(vg_coco_data.intersection(coco_train_ids)))

#okay, so the ids are kind of mixed up, but that is okay, we only need to use the train ids now
'''
