%env IMAGE_DIR = images

mkdir $IMAGE_DIR
cd $IMAGE_DIR

wget http://images.cocodataset.org/zips/test2017.zip

unzip test2017.zip


cd ../

wget http://images.cocodataset.org/annotations/image_info_test2017.zip

unzip image_info_test2017.zip


