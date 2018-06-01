# Age Progression/Regression by Conditional Adversarial Autoencoder (CAAE)

TensorFlow implementation of the algorithm in the paper [Age Progression/Regression by Conditional Adversarial Autoencoder](http://web.eecs.utk.edu/~zzhang61/docs/papers/2017_CVPR_Age.pdf).

<p align="center">
  <img src="demo/method.png" width="500">
</p>

## Datasets
* FGNET
* [MORPH](https://ebill.uncw.edu/C20231_ustores/web/product_detail.jsp?PRODUCTID=8)
* [CACD](http://bcsiriuschen.github.io/CARC/)
* UTKFace (Access from the [Github link](https://susanqq.github.io/UTKFace/) or the [Wiki link](http://aicip.eecs.utk.edu/wiki/UTKFace))

## Custom Training
```
$ python main.py
    --dataset		default 'UTKFace'. Please put your own dataset in ./data
    --savedir		default 'save'. Please use a meaningful name, e.g., save_init_model.
    --epoch		default 50.
    --use_trained_model	default True. If use a trained model, savedir specifies the model name. 
    --use_init_model	default True. If load the trained model failed, use the init model save in ./init_model 
```

## Testing
```
$ python main.py --is_train False --testdir your_image_dir --savedir save
```
**Note**: `savedir` specifies the model name saved in the training. By default, the trained model is saved in the folder save (i.e., the model name).
Then, it is supposed to print out the following message.

```
  	Building graph ...

	Testing Mode

	Loading pre-trained model ...
	SUCCESS ^_^

	Done! Results are saved as save/test/test_as_xxx.png
```

Specifically, the testing faces will be processed twice, being considered as male and female, respectively. Therefore, the saved files are named `test_as_male.png` and `test_as_female.png`, respectively. To achieve better results, it is necessary to train on a large and diverse dataset.

## Files
* [`FaceAging.py`](FaceAging.py) is a class that builds and initializes the model, and implements training and testing related stuff
* [`ops.py`](ops.py) consists of functions called `FaceAging.py` to implement options of convolution, deconvolution, fully connection, leaky ReLU, load and save images.   
* [`main.py`](main.py) demonstrates `FaceAging.py`.

