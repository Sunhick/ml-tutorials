1. http://www.deeplearningbook.org/

2. http://people.idsia.ch/~juergen/

3. http://cs231n.github.io/

My Lecture Slides: I will download and share it in this. Its secured so.

---

* http://www.jennwv.com/courses/F11.html
* http://alex.smola.org/teaching/cmu2013-10-701/slides/
* http://www.cs.cmu.edu/~ninamf/courses/

statistics
* http://vassarstats.net/textbook/


An Overview of the Second Chapter of Deep Learning and Exercise:


http://neuralnetworksanddeeplearning.com/chap6.html Mandatory to read everything till " ... in Practice" Also interesting is this part: “The Code for Our Concolutional Networks” From “The Recent Progress…” it is only for information (not mandatory now)

Another lecture for better understanding: http://videolectures.net/machine_krizhevsky_imagenet_classification/

Nice Tutorial and Wrap-Up after the first Exercise (Introducing also LeNet with architecture information and graphics): https://shadowthink.com/blog/tech/2016/08/28/Caffe-MNIST-tutorial

Original Website by LeCun http://yann.lecun.com/exdb/lenet/

I also like: https://www.youtube.com/watch?v=bEUX_56Lojc but it is a bit difficult for people with less math background

Exercise: a) train AlexNet for Object Recognition (CIFAR-10 Dataset) --- The dataset can be downloaded from the following website, https://www.cs.toronto.edu/~kriz/cifar.html It is very easy to read the data in python Please go to the section “Dataset layout” and you will find the python code to read it.

Note. The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch. Each of these files is a Python "pickled" object produced with cPickle. Here is a Python routine which will open such a file and return a dictionary: It has keys ‘data’, ‘labels’ and ‘label_names’. The following is copied from website data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image. labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data. For intuitive names following array could be used label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc. Following is the sample code for writing the images to disk [[Media:exercise2.py]]
b) change the architecture (depth of the FFN, or remove a ConvNet Block, use other activation functions)

c) change it as you think it might improve the recognition




