### NHANES II

Unfortunately since the end of January the website[(https://healthdata.gov/dataset/Digitized-NHANES-II-X-ray-Films/f2q3-ewha/about_data)] hosting this dataset has been taken offline. If this changes please let me know and I can update this page.

### CSXA

The CSXA images and labels can be accessed here [(https://www.scidb.cn/en/detail?dataSetId=8e3b3d5e60a348ba961e19d48b881c90)]. The images should be able to be passed into the model as is. However, to evaluate the results versus the ground truth, for each vertebra you must convert set of corners into a bitmap. I cannot find the code that I used for this, but it shouldn't be too difficult to recreate. 

Just to note, when training the models I only used a small subset of the dataset (200 train, 200 val, 200 test). This was mainly to save on compute. Its also worth noting that some of the later images in the dataset appear to have inverted colours, these images were excluded from my study.

### Data Split

I've included here the IDs of the train / val / test samples.