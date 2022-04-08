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

/****4  training****/
python train.py --cfg_file cfgs/da-nuscenes-kitti_models/pvrcnn/pvrcnn_old_anchor.yaml --batch_size 1


/****5 training pretrained ****/
bash scripts/dist_train.sh 1 --cfg_file cfgs/da-nuscenes-kitti_models/pvrcnn/pvrcnn_old_anchor.yaml --batch_size 1


/****6 self-training process****/
bash scripts/dist_train.sh 1 --cfg_file cfgs/da-nuscenes-kitti_models/pvrcnn_st3d/pvrcnn_st3d.yaml --batch_size 4 --pretrained_model '/home/algo-4/work/ST3D/tools/pvrcnn_st3d_ckpt.pth'


/****   2 pandaset to kitti        ****/
/****Dataset Preparation dataloader****/
#pandaset dataloader
python -m pcdet.datasets.pandaset.pandaset_dataset create_pandaset_infos tools/cfgs/dataset_configs/pandaset_dataset.yaml

/****************************spconv1********************************************/
create -n spconv1 python=3.6 pytorch=1.1 torchvision cudatoolkit=9.2

git clone https://github.com/traveller59/spconv.git --recursive
cd spconv/
git checkout 8da6f96


cd third_party/
git clone https://github.com/pybind/pybind11.git
cd pybind11/
git checkout 085a294

cd ../..
python setup.py bdist_wheel

