---
layout: post
title: Image segmentation with a U-Net-like architecture
subtitle: keras example - Segmentation 
author: san9hyun
categories: keras-example
banner : /assets/images/banners/solar-system.jpg
tags: U-Net Segmentation 
---
**dark 모드로 보기** <br>



## 데이터 다운로드


```python
#curl로 데이터 다운로드 하였을 때, gzip: stdin: not in gzip format 에러발생

!wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
!tar -xf images.tar.gz
!tar -xf annotations.tar.gz
```

    --2022-12-05 07:02:18--  http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    Resolving www.robots.ox.ac.uk (www.robots.ox.ac.uk)... 129.67.94.2
    Connecting to www.robots.ox.ac.uk (www.robots.ox.ac.uk)|129.67.94.2|:80... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz [following]
    --2022-12-05 07:02:18--  https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    Connecting to www.robots.ox.ac.uk (www.robots.ox.ac.uk)|129.67.94.2|:443... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz [following]
    --2022-12-05 07:02:19--  https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
    Resolving thor.robots.ox.ac.uk (thor.robots.ox.ac.uk)... 129.67.95.98
    Connecting to thor.robots.ox.ac.uk (thor.robots.ox.ac.uk)|129.67.95.98|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 791918971 (755M) [application/octet-stream]
    Saving to: ‘images.tar.gz’
    
    images.tar.gz       100%[===================>] 755.23M  26.1MB/s    in 29s     
    
    2022-12-05 07:02:48 (26.1 MB/s) - ‘images.tar.gz’ saved [791918971/791918971]
    
    --2022-12-05 07:02:48--  http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
    Resolving www.robots.ox.ac.uk (www.robots.ox.ac.uk)... 129.67.94.2
    Connecting to www.robots.ox.ac.uk (www.robots.ox.ac.uk)|129.67.94.2|:80... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz [following]
    --2022-12-05 07:02:48--  https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
    Connecting to www.robots.ox.ac.uk (www.robots.ox.ac.uk)|129.67.94.2|:443... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz [following]
    --2022-12-05 07:02:49--  https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz
    Resolving thor.robots.ox.ac.uk (thor.robots.ox.ac.uk)... 129.67.95.98
    Connecting to thor.robots.ox.ac.uk (thor.robots.ox.ac.uk)|129.67.95.98|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 19173078 (18M) [application/octet-stream]
    Saving to: ‘annotations.tar.gz’
    
    annotations.tar.gz  100%[===================>]  18.28M  14.4MB/s    in 1.3s    
    
    2022-12-05 07:02:51 (14.4 MB/s) - ‘annotations.tar.gz’ saved [19173078/19173078]



## 입력 이미지와 타겟 Segmentation masks 경로 준비


```python
import os

input_dir = "images/"
target_dir = "annotations/trimaps/"
img_size = (160,160)
num_classes = 3
batch_size = 32
```


```python
input_img_paths = sorted(
    [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".jpg")]
)

target_img_paths = sorted(
    [os.path.join(target_dir, fname) for fname in os.listdir(target_dir) if fname.endswith(".png") and not fname.startswith(".")]
)

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:3], target_img_paths[:3]):
  print(input_path,"|", target_path)
```

    Number of samples: 7390
    images/Abyssinian_1.jpg | annotations/trimaps/Abyssinian_1.png
    images/Abyssinian_10.jpg | annotations/trimaps/Abyssinian_10.png
    images/Abyssinian_100.jpg | annotations/trimaps/Abyssinian_100.png


## 입력 이미지와 Segmentation mask 예시 확인


```python
from IPython.display import Image, display
from tensorflow import keras
from keras.utils import load_img                            
from PIL import ImageOps
```


```python
# 입력이미지 예시
display(Image(filename=input_img_paths[0]))
```


![jpeg](/assets/images/contents/keras-examples/Image segmentation with a U-Net-like architecture/Image segmentation with a U-Net-like architecture_7_0.jpg)


```python
#segmentation mask 예시

img = ImageOps.autocontrast(load_img(target_img_paths[0]))                        #autocontrast : 가장 밝은 픽셀은 흰색(255) 가장 어두운 픽셀은 검은색(0)이 되도록 이미지 매핑
display(img)
```


![png](/assets/images/contents/keras-examples/Image segmentation with a U-Net-like architecture/Image segmentation with a U-Net-like architecture_8_0.png)


## 배치 데이터를 로드하고 벡터화하는 클래스 준비


```python
import numpy as np
```


```python
class OxfordPets(keras.utils.Sequence):

  def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
    self.batch_size = batch_size
    self.img_size = img_size
    self.input_img_paths = input_img_paths
    self.target_img_paths = target_img_paths

  def __len__(self):
    return len(self.target_img_paths) // self.batch_size
  
  def __getitem__(self, idx):
    i = idx*self.batch_size                                                      #idx:배치의 인덱스 
    batch_input_img_paths = self.input_img_paths[i : i+self.batch_size]          
    batch_target_img_paths = self.target_img_paths[i : i+self.batch_size]
    
    x = np.zeros((self.batch_size, ) + self.img_size + (3,), dtype="float32")    #(batch_size,160,160,3) 
    for j, path in enumerate(batch_input_img_paths):
      img = load_img(path, target_size=self.img_size)
      x[j] = img                                                                 #(160,160,3) 

    y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")       #(batch_size,160,160,1)
    for j,path in enumerate(batch_target_img_paths):
      img = load_img(path, target_size=self.img_size, color_mode="grayscale")    #(160,160)
      y[j] = np.expand_dims(img,2)                                               #(160,160,1)
      y[j] -=1                                                                   #실측 레이블은 1,2,3. -> 0,1,2로 변경 why? (1:Foreground, 2:Background, 3:Unknown)


    return x, y
```


```python
img = load_img(target_img_paths[0],target_size=(160,160) ,color_mode="grayscale")
```


```python
from keras import layers
```


```python
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

  
    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x                                                # 잔차학습을 위해 따로 빼놓기

   
    ### [First half of the network: downsampling inputs] ###
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )

        x = layers.add([x, residual]) 
        previous_block_activation = x  

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

model = get_model(img_size, num_classes)
keras.utils.plot_model(model, show_shapes=True)
```




![png](/assets/images/contents/keras-examples/Image segmentation with a U-Net-like architecture/Image segmentation with a U-Net-like architecture_14_0.png)



## 검증데이터셋 준비


```python
import random

val_samples = 1000
random.Random(1337).shuffle(input_img_paths)                                     # 이런방법이 있구나..
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]
```


```python
train_gen = OxfordPets(batch_size, img_size, train_input_img_paths, target_img_paths)
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
```

## 모델 학습


```python
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")       #target이 정수이기 때문에 "sparse"

callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
]

epochs = 15
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
```

    Epoch 1/15
    230/230 [==============================] - 75s 266ms/step - loss: 0.9377 - val_loss: 1.3684
    Epoch 2/15
    230/230 [==============================] - 60s 259ms/step - loss: 0.5548 - val_loss: 0.9865
    Epoch 3/15
    230/230 [==============================] - 58s 253ms/step - loss: 0.4825 - val_loss: 0.4573
    Epoch 4/15
    230/230 [==============================] - 58s 253ms/step - loss: 0.4428 - val_loss: 0.4153
    Epoch 5/15
    230/230 [==============================] - 58s 252ms/step - loss: 0.4109 - val_loss: 0.4291
    Epoch 6/15
    230/230 [==============================] - 59s 257ms/step - loss: 0.3881 - val_loss: 0.3722
    Epoch 7/15
    230/230 [==============================] - 58s 251ms/step - loss: 0.3633 - val_loss: 0.4071
    Epoch 8/15
    230/230 [==============================] - 58s 249ms/step - loss: 0.3445 - val_loss: 0.3635
    Epoch 9/15
    230/230 [==============================] - 57s 246ms/step - loss: 0.3288 - val_loss: 0.4315
    Epoch 10/15
    230/230 [==============================] - 58s 253ms/step - loss: 0.3151 - val_loss: 0.3504
    Epoch 11/15
    230/230 [==============================] - 57s 248ms/step - loss: 0.3010 - val_loss: 0.3940
    Epoch 12/15
    230/230 [==============================] - 57s 246ms/step - loss: 0.2921 - val_loss: 0.3923
    Epoch 13/15
    230/230 [==============================] - 56s 244ms/step - loss: 0.2821 - val_loss: 0.3986
    Epoch 14/15
    230/230 [==============================] - 57s 248ms/step - loss: 0.2738 - val_loss: 0.3692
    Epoch 15/15
    230/230 [==============================] - 56s 243ms/step - loss: 0.2661 - val_loss: 0.4219





    <keras.callbacks.History at 0x7fe01007b5e0>




```python
## 예측 시각화
```


```python
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
val_preds = model.predict(val_gen)


def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    display(img)


# 검증용 이미지
i = 10

# 검증용 이미지 확인
display(Image(filename=val_input_img_paths[i]))

# 검증용이미지 target 확인
img = ImageOps.autocontrast(load_img(val_target_img_paths[i]))
display(img)

# 모델에 의해 예측된 mask 확인
display_mask(i)  
```

    31/31 [==============================] - 6s 167ms/step



![jpeg](/assets/images/contents/keras-examples/Image segmentation with a U-Net-like architecture/Image segmentation with a U-Net-like architecture_21_1.jpg)



![png](/assets/images/contents/keras-examples/Image segmentation with a U-Net-like architecture/Image segmentation with a U-Net-like architecture_21_2.png)



![png](/assets/images/contents/keras-examples/Image segmentation with a U-Net-like architecture/Image segmentation with a U-Net-like architecture_21_3.png)



```python
#### 마크다운 파일로 저장
!jupyter nbconvert --to markdown "/content/drive/MyDrive/Colab Notebooks/keras-example/computer vision/image classification from scratch/image_classification_from_scratch.ipynb"
```

    [NbConvertApp] WARNING | pattern '/content/drive/MyDrive/Colab Notebooks/keras-example/computer vision/Image segmentation with a U-Net-like architecture/Image segmentation with a U-Net-like architecture.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only 
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place, 
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document. 
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides']
                or a dotted object name that represents the import path for an
                `Exporter` class
        Default: 'html'
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template file to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_file]
    --writer=<DottedObjectName>
        Writer class used to write the 
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        overwrite base name use for output files.
                    can only be used when converting one notebook at a time.
        Default: ''
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current 
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy 
                of reveal.js. 
                For speaker notes to work, this must be a relative path to a local 
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb
    
                which will convert mynotebook.ipynb to the default format (probably HTML).
    
                You can specify the export format with `--to`.
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic' and 'full'. You
                can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template basic mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of 
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.




```python

```