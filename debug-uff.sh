DATA_DIR=$HOME/Downloads/new-tests
# cam0_27.png
# cam0_59.png
# cam2_21.png
# cam3_107.png
# cam3_146.png
# cam3_148.png
# cam3_52.png
# cam3_63.png

test_vgg_model(){
    ./uff-runner.py --base-model=vgg --path-to-npz=$HOME/Downloads/vgg450000_no_cpm.npz --image=$1
}

test_vgg_model ${DATA_DIR}/acam0_27.png  # UFF has no result at all
# test_vgg_model ./data/media/COCO_val2014_000000000192.jpg  # UFF has worse result than TF

test_hao28_model(){
    ./uff-runner.py --base-model=hao28 --path-to-npz=$HOME/Downloads/hao28/pose345000.npz --image=$1
}


# test_hao28_model ./data/media/COCO_val2014_000000000192.jpg  # UFF has no result
# test_hao28_model ${DATA_DIR}/cam0_27.png  # UFF has no result
