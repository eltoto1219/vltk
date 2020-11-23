from transformers import LxmertForPreTraining, LxmertConfig


#for bounding boxes, lets just make them zero --> to save time ofcourse
config = LxmertConfig(
  visual_attr_loss=False,
  visual_feat_loss=True,
  visual_obj_loss=False,
)

# okay this is the actual lxmert pretraining model
lxmert_pretraining = LxmertForPreTraining(config)

#we can just init the bounding boxes to zero --> specify in the dataloader
#next, we just need to actually extract all of the lxmert data
#okay, that shouldnt be to hard, all I need to do is download from mscoco + visual genome



