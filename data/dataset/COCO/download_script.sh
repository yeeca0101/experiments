%env IMAGE_DIR = images

mkdir $IMAGE_DIR
cd $IMAGE_DIR

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip

upzip test2017.zip
unzip train2017.zip
unzip val2017.zip

rm train2017.zip
rm val2017.zip

cd ../

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip

unzip image_info_test2017.zip
unzip annotations_trainval2017.zip
unzip stuff_annotations_trainval2017.zip

rm image_info_test2017.zip
rm annotations_trainval2017.zip
rm stuff_annotations_trainval2017.zip

