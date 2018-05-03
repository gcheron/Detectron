import pickle

net2finetune = '/sequoia/data2/gcheron/detectron/detectron-download-cache/36228933/12_2017_baselines/fast_rcnn_R-101-FPN_2x.yaml.09_26_27.jkOUTrrk/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl'
respath = '/sequoia/data2/gcheron/detectron/detectron-download-cache/36228933/12_2017_baselines/fast_rcnn_R-101-FPN_2x.yaml.09_26_27.jkOUTrrk/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_for_FT.pkl'


with open('/sequoia/data2/gcheron/detectron/detectron-download-cache/ImageNetPretrained/MSRA/R-101.pkl', 'r') as f:
   baseline = pickle.load(f)

with open(net2finetune, 'r') as f:
   needFinetune = pickle.load(f)
needFinetune = needFinetune['blobs'] # remove cfg field

# check file to fine tune has all fields
for i in baseline:
   assert i in needFinetune, '%s is missing' % i

tosave = {}
# keep only the same fields
for i in needFinetune:
   if i in baseline:
      print i
      tosave[i] = needFinetune[i]

with open(respath, 'wb') as f:
   pickle.dump(tosave, f)

print 'model ready to fine tune in:\n%s' % respath
