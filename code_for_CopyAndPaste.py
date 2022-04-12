/****************************ST3D********************************************/
/****   1 nuscense to kitti        ****/
/****Dataset Preparation dataloader****/
#kitti dataloader
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml

#nuscenes dataloader
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos     --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml     --version v1.0-mini

/****2 testing****/
python test.py --cfg_file cfgs/da-nuscenes-kitti_models/pvrcnn_st3d/pvrcnn_st3d.yaml --batch_size 2 --ckpt pvrcnn_st3d_ckpt.pth --eval_all

/****3 demo visualization****/
python demo.py --cfg_file cfgs/da-nuscenes-kitti_models/pvrcnn_st3d/pvrcnn_st3d.yaml \
    --ckpt pvrcnn_st3d_sn_ckpt.pth \
    --data_path '/home/algo-4/work/ST3D/data/kitti/testing/velodyne/000004.bin'

/****4 Pre-training with ROS****/
python train.py --cfg_file '/home/algo-4/work/ST3D/tools/cfgs/da-nuscenes-kitti_models/pvrcnn/pvrcnn_old_anchor_ros.yaml'

/****5 training pretrained ****/
bash scripts/dist_train.sh 1 --cfg_file cfgs/da-nuscenes-kitti_models/pvrcnn/pvrcnn_old_anchor.yaml --batch_size 1


/****6 self-training process****/
bash scripts/dist_train.sh 1 --cfg_file cfgs/da-nuscenes-kitti_models/pvrcnn_st3d/pvrcnn_st3d.yaml --batch_size 4 --pretrained_model '/home/algo-4/work/ST3D/tools/pvrcnn_st3d_ckpt.pth'


/****   2 pandaset to kitti        ****/
/****Dataset Preparation dataloader****/
#pandaset dataloader
python -m pcdet.datasets.pandaset.pandaset_dataset create_pandaset_infos tools/cfgs/dataset_configs/pandaset_dataset.yaml

/****2 missing****/
due to lack of evalutation mertics

/****3 demo visualization****/
 python demo2.py --cfg_file cfgs/pandaset_models/pv_rcnn.yaml --ckpt pandaset-pv-rcnn_checkpoint_epoch_80.pth --data_path '/home/algo-4/work/ST3D/data/pandaset/004/lidar/02.pkl.gz'


/****4  training with old anchor ****/
/***4.1 secondiou ****/
python train.py --cfg_file cfgs/da-pandaset-kitti_models/secondiou/secondiou_old_anchor.yaml --batch_size 4

/***4.2 PVRCNN  ****/
python train.py  --cfg_file cfgs/da-pandaset-kitti_models/pvrcnn/pvrcnn_old_anchor.yaml --batch_size 2


error : an illegal memory access was encountered


/*ok*/ train with pvrcnn on pandaset
python train.py --cfg_file cfgs/pandaset_models/pv_rcnn.yaml  --batch_size 1 


/****************************spconv1********************************************/
create -n spconv1 python==3.6 pytorch=1.2 torchvision cudatoolkit=9.2

!pip install torch==1.2 torchvison==0.4 
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv/
git checkout 8da6f96


cd third_party/
git clone https://github.com/pybind/pybind11.git
cd pybind11/
git checkout 085a294

cd ../..
python setup.py bdist_wheel

cd ./dist && pip install xxx.whl to install spconv.

/***************OpenPCDet**********************/
python demo.py --cfg_file cfgs/pandaset_models/pv_rcnn.yaml --ckpt pv-rcnn_checkpoint_epoch_80.pth --data_path /home/algo-4/Desktop/txt2npy_original_ground_revert.npy


python demo2.py --cfg_file cfgs/pandaset_models/pv_rcnn.yaml --ckpt pandaset-pv-rcnn_checkpoint_epoch_80.pth --data_path /home/algo-4/work/OpenPCDet/data/Pandaset/004/lidar/02.pkl.gz


/*************deal with torch 1.2 version .pth cant load problem******************/
state_dict = torch.load("xxx.pth")
torch.save(state_dict, "xxx.pth", _use_new_zipfile_serialization=False)


/*************kill cuda memory******************/
sudo fuser -v /dev/nvidia*
sudo kill -9 PID.


/** trouble shooting **/
dataloader drop_last
# data length cant divide batchsize, drop last batch