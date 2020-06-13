### How does human face mask detector work?
 - ![explanaition](screen.png)

 - Result of human face detector
 ![https://d3fa68hw0m2vcc.cloudfront.net/563/195539828.jpeg](mask-ouput.png)
![https://image.freepik.com/free-photo/african-american-young-volunteer-woman-wearing-face-mask-outdoors-coronavirus-quarantine-global-pandemic_151355-5475.jpg](mask_screenshot.png)
![https://minds-africa.org/wp-content/uploads/2019/11/sellasie-1038x890.jpg](withoutmask_screenshot.png)

reference for images: ![First Pictures](https://d3fa68hw0m2vcc.cloudfront.net/563/195539828.jpeg), ![second Picture](https://image.freepik.com/free-photo/african-american-young-volunteer-woman-wearing-face-mask-outdoors-coronavirus-quarantine-global-pandemic_) and ![Third Picture](https://minds-africa.org/wp-content/uploads/2019/11/sellasie-1038x890.jpg)
- 
 - 
 - 

### Install dependencies
 - [x] tensorflow
 - [x] numpy      
 - [x] sklearn  (scikit-learn) 
 - [x] tensorflow-gpu
 - [x] matplotlib
 - [x] imutils

### Clone the project
```
$git clone https://github.com/bm777/humanface-mask-detector.git
cd humanface-mask-detector
```

### How to implement your own face mask detector

 - Fine tuning
 Load model: res10_300x300_ssd_iter_140000.caffemodel.
 We Construct  new FC head(Fully connected layers), append it the base model and we freeze the base layers of the network(res10.*.caffemodel)
 -Notice: the weight sof the layers(base model) will not uploaded during the process of backpropagation. Whereas the head layer weight will be tunued. 
 - Training, testing and validation stage
 In this section, we load face mask dataset from disk and then train a model using tensorflow.keras 2.1.0.
 To perform your accuracy, you can change your hyperparameters(batch size and number of epoch learning rate)
 ```
 python3 fm_train.py --dataset dataset
 ```
![Training phase](best-accu.png)
 - History ploted

![History of training and validation stage](ploted.png)

### Credit
 - the dataset was created by : [Prajna Bhandary](https://lnkd.in/fJTAP_D)
 - part of code from [Adrian Rosebrock](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)
 - part of fine tuning from this : [LINK](https://www.pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/)
 - part of face Landmark to augment database: ALready done by Adrian Rosebrock
 - pretrained model used: MobileNetV2 -> [ImageNet weights](http://www.image-net.org/)