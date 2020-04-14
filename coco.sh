wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
7z x train2017.zip -odatasets/coco/images
7z x val2017.zip -odatasets/coco/images
7z x annotations_trainval2017.zip -odatasets/coco