import json
import pickle
import ipdb
import re
import numpy as np


def get_UCF101(setname):
   datasetroot = '/sequoia/data2/gcheron/UCF101'
   class_list = ('Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling',
                     'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing',
                     'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing',
                     'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping',
                     'VolleyballSpiking', 'WalkingWithDog')
   gtpath = datasetroot + '/detection/gtfile.pkl'

   resroot = datasetroot + '/detection/instances_%s1_ucf101' % setname
   if setname == 'val':
      setname = 'test'
   trainlist = datasetroot + '/detection/OF_vidlist_%s1.txt' % setname

   onkeyframes = False
   check_WH = False

   return class_list, gtpath, trainlist, onkeyframes, check_WH, resroot


def get_DALY(setname):
   datasetroot = '/sequoia/data2/gcheron/DALY'
   class_list = ('ApplyingMakeUpOnLips', 'BrushingTeeth', 'CleaningFloor', 'CleaningWindows', 'Drinking',
                     'FoldingTextile', 'Ironing', 'Phoning', 'PlayingHarmonica', 'TakingPhotosOrVideos')
   gtpath = datasetroot + '/gtfile.pkl'

   resroot = datasetroot + '/instances_%s1_daly' % setname
   if setname == 'val':
      setname = 'test'
   trainlist = datasetroot + '/OF_vidlist_%s1.txt' % setname

   onkeyframes = True
   check_WH = True


   return class_list, gtpath, trainlist, onkeyframes, check_WH, resroot


def get_gt_instance_annot(inst, onkeyframes, nkeyframes):
   if onkeyframes:
      boxes = []
      frames = []
      classes = []
      for keyframe in inst['keyframes']['keylist']:
         box = keyframe['boxes'][None, :]
         assert box.shape == (1, 4)
         boxes.append(box)

         frames.append(keyframe['frame_num'])
      boxes = np.concatenate(boxes)

   else:
      boxes = inst['boxes']
      tbound = inst['tbound']
      t0 = tbound[0]
      flen = tbound[1] - tbound[0] + 1
      frames = [f + t0 for f in range(flen)]

   classes = [inst['label']] * len(frames) # start at 1

   # boxes start at 0
   boxes = boxes - 1
   # boxes are in x1, y1, W, H format
   W = boxes[:, 2] - boxes[:, 0] + 1
   H = boxes[:, 3] - boxes[:, 1] + 1
   boxes[:, 2] = W
   boxes[:, 3] = H

   areas = W * H

   assert len(frames) == boxes.shape[0] and boxes.shape[1] == 4

   keyframes = {'inst': inst}
   if nkeyframes > 0:
      perm = np.random.permutation(len(frames))
      keep = perm[:nkeyframes]
      boxes = boxes[keep, :]
      frames = np.array(frames)[keep].tolist()
      classes = np.array(classes)[keep].tolist()
      areas = areas[keep]
      keyframes['boxes'] = boxes
      keyframes['frames'] = frames
      keyframes['idx'] = keep

   return boxes, frames, classes, areas, keyframes

# coco example
#cocodataset = json.load(open('/sequoia/data2/jpeyre/datasets/coco/annotations/instances_train2014.json', 'r'))
np.random.seed(19)


setname = 'train' # train | val
dataset_name = 'ucf101' # ucf101 | daly
nkeyframes = -1 # -1 | 1 | 3

if dataset_name == 'ucf101':
   get_fun = get_UCF101
elif dataset_name == 'daly':
   get_fun = get_DALY
else:
   raise ValueError

class_list, gtpath, trainlist, onkeyframes, check_WH, resroot = get_fun(setname)

respath = resroot
if nkeyframes > 0:
   respath += '_keyframes%d' % (nkeyframes)
reskeylist = respath + '_keylist.pkl'
respath += '.json'

# get ground truth
with open(gtpath) as f:
   gtfile = pickle.load(f) 

# get video list
with open(trainlist) as f:
   vcontent = f.readlines()
vidlist = [re.sub(' .*', '', x.strip()) for x in vcontent]


dataset = {'categories': [], 'annotations': [], 'images': []}
# define class list
for c, class_name in enumerate(class_list):
  dataset['categories'].append({'id': c+1, 'name': class_name}) # start at 1 


imname2id = {}
numinstance = 0
all_keyframes = {}
for v, vid in enumerate(vidlist): # for all videos from the split
   # fill the following:
   # annotations: ['area', 'image_id', 'bbox', 'category_id', 'id']
   # images: ['file_name', 'height', 'width', 'id']

   if v % 10 == 0:
      print '%d out of %d videos (%s)' % (v, len(vidlist), vid)

   # get video gt
   vid_gt = gtfile[vid]
   all_keyframes[vid] = {'instances': [], 'allinfo': vid_gt}

   vlen = vid_gt['length']

   if check_WH:
      imW, imH = vid_gt['WH_size']
   else:
      imW, imH = 320, 240


   for i_inst, inst in enumerate(vid_gt['gts']): # for all video GT instances
      numinstance += 1
      boxes, frames, classes, areas, keyframes = get_gt_instance_annot(inst, onkeyframes, nkeyframes)

      keyframes['inst_id'] = i_inst
      all_keyframes[vid]['instances'].append(keyframes)

      for i, f in enumerate(frames):
         frame_name = '%s/image-%05d.jpg' % (vid, f)

         if not frame_name in imname2id:
            # add new image
            imname2id[frame_name] = len(imname2id)
            iminfo = {'file_name': frame_name, 'height': imH, 'width': imW, 'id': imname2id[frame_name]}
            dataset['images'].append(iminfo)            

         # add new annotation
         f_id = imname2id[frame_name]
         bbox = boxes[i]
         area = float(areas[i])
         aid = len(dataset['annotations'])
      
         anninfo = {'image_id': f_id, 'bbox': bbox.tolist(), 'category_id': classes[i], 'id': aid, 'area': area,
                    'segmentation': [[-1]], 'iscrowd': False}
         dataset['annotations'].append(anninfo)

with open(respath, 'w') as f:
   json.dump(dataset, f)

if nkeyframes > 0:
   with open(reskeylist, 'w') as f:
      pickle.dump(all_keyframes, f)

print 'Saved %d annotations (on %d images and %d instances) to:\n%s' % (len(dataset['annotations']), len(dataset['images']), numinstance, respath)
