```python
import os
import zipfile
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from datasets import load_dataset
import os
import shutil
from PIL import Image
import glob
```

    2025-05-16 09:55:38.368722: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2025-05-16 09:55:38.378668: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1747368638.390058    9397 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1747368638.393805    9397 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    W0000 00:00:1747368638.402638    9397 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1747368638.402646    9397 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1747368638.402648    9397 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1747368638.402649    9397 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    2025-05-16 09:55:38.405417: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.



```python
from huggingface_hub import notebook_login

notebook_login()
```


    VBox(children=(HTML(value='<center> <img\nsrc=https://huggingface.co/front/assets/huggingface_logo-noborder.sv…



```python
data = load_dataset("MadanKhatri/house_problem_images")
```


```python
ex = data['train'][100]
ex
```




    {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=686x386>,
     'label': 3}




```python
image = ex['image']
image
```




    
![png](output_4_0.png)
    




```python
labels = data['train'].features['label']
labels
```




    ClassLabel(names=['builder', 'electrician', 'others', 'plumber'], id=None)




```python
labels.int2str(ex['label'])
```




    'plumber'




```python
from evaluate import load

metric = load("accuracy")
```


```python
data
```




    DatasetDict({
        train: Dataset({
            features: ['image', 'label'],
            num_rows: 1758
        })
        test: Dataset({
            features: ['image', 'label'],
            num_rows: 311
        })
    })




```python
labels = data["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label
```


```python
from transformers import ViTFeatureExtractor

model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
```

    /home/madan/.pyenv/versions/pyenvs/lib/python3.10/site-packages/transformers/models/vit/feature_extraction_vit.py:28: FutureWarning: The class ViTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use ViTImageProcessor instead.
      warnings.warn(



```python
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size['height']), # Access the height or width key
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(feature_extractor.size['height']), # Access the height or width key
            CenterCrop(feature_extractor.size['height']), # Access the height or width key
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch
```


```python
train_ds = data['train']
val_ds = data['test']
```


```python
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)
```


```python
train_ds[0]
```




    {'image': <PIL.Image.Image image mode=RGB size=3850x2891>,
     'label': 1,
     'pixel_values': tensor([[[-0.4118, -0.4118, -0.4196,  ..., -0.9529, -0.9529, -0.8745],
              [-0.4118, -0.4039, -0.4118,  ..., -0.9451, -0.9451, -0.9294],
              [-0.4118, -0.4039, -0.3961,  ..., -0.9451, -0.9608, -0.9686],
              ...,
              [-0.4745, -0.4196, -0.4196,  ..., -0.3412, -0.4118, -0.3020],
              [-0.4275, -0.4118, -0.4667,  ..., -0.2706, -0.3490, -0.1765],
              [-0.3804, -0.4118, -0.5216,  ..., -0.2157, -0.2000, -0.1843]],
     
             [[-0.4510, -0.4588, -0.4588,  ..., -0.9608, -0.9529, -0.8902],
              [-0.4510, -0.4588, -0.4510,  ..., -0.9529, -0.9529, -0.9373],
              [-0.4510, -0.4510, -0.4510,  ..., -0.9529, -0.9686, -0.9765],
              ...,
              [-0.5059, -0.4588, -0.4667,  ..., -0.3647, -0.4275, -0.3333],
              [-0.4588, -0.4588, -0.5137,  ..., -0.3098, -0.3804, -0.2235],
              [-0.4196, -0.4431, -0.5451,  ..., -0.2627, -0.2392, -0.2392]],
     
             [[-0.5137, -0.5216, -0.5059,  ..., -0.9765, -0.9608, -0.8980],
              [-0.5216, -0.5216, -0.5059,  ..., -0.9765, -0.9686, -0.9451],
              [-0.5216, -0.5216, -0.5137,  ..., -0.9843, -0.9843, -0.9843],
              ...,
              [-0.5529, -0.5059, -0.5137,  ..., -0.4196, -0.4902, -0.3882],
              [-0.5059, -0.5059, -0.5529,  ..., -0.3569, -0.4431, -0.2706],
              [-0.4667, -0.4902, -0.5922,  ..., -0.3176, -0.3020, -0.2941]]])}




```python
from transformers import ViTForImageClassification
model_name_or_path = 'google/vit-base-patch16-224-in21k'
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)
```

    Some weights of ViTForImageClassification were not initialized from the model checkpoint at google/vit-base-patch16-224-in21k and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='finetuned-occupations',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="steps",
    num_train_epochs=10,  # Updated from 9 to 10
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
    report_to='tensorboard',
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='./logs',
    seed=42
)
```


```python
import numpy as np

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
```


```python
import torch

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }
```


```python
from transformers import Trainer
trainer = Trainer(
    model,
    training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
```

    /tmp/ipykernel_9397/963735583.py:2: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
      trainer = Trainer(



```python
train_results = trainer.train()
# rest is optional but nice to have
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
```

    /home/madan/.pyenv/versions/pyenvs/lib/python3.10/site-packages/PIL/Image.py:1043: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
      warnings.warn(




    <div>

      <progress value='1100' max='1100' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [1100/1100 08:58, Epoch 10/10]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>100</td>
      <td>0.589300</td>
      <td>0.514178</td>
      <td>0.842444</td>
    </tr>
    <tr>
      <td>200</td>
      <td>0.424800</td>
      <td>0.370143</td>
      <td>0.868167</td>
    </tr>
    <tr>
      <td>300</td>
      <td>0.247400</td>
      <td>0.361343</td>
      <td>0.884244</td>
    </tr>
    <tr>
      <td>400</td>
      <td>0.178300</td>
      <td>0.363787</td>
      <td>0.881029</td>
    </tr>
    <tr>
      <td>500</td>
      <td>0.129700</td>
      <td>0.382830</td>
      <td>0.897106</td>
    </tr>
    <tr>
      <td>600</td>
      <td>0.151200</td>
      <td>0.393535</td>
      <td>0.900322</td>
    </tr>
    <tr>
      <td>700</td>
      <td>0.085600</td>
      <td>0.382469</td>
      <td>0.903537</td>
    </tr>
    <tr>
      <td>800</td>
      <td>0.061900</td>
      <td>0.408207</td>
      <td>0.900322</td>
    </tr>
    <tr>
      <td>900</td>
      <td>0.056000</td>
      <td>0.346442</td>
      <td>0.900322</td>
    </tr>
    <tr>
      <td>1000</td>
      <td>0.063600</td>
      <td>0.332707</td>
      <td>0.919614</td>
    </tr>
    <tr>
      <td>1100</td>
      <td>0.051900</td>
      <td>0.320150</td>
      <td>0.913183</td>
    </tr>
  </tbody>
</table><p>


    /home/madan/.pyenv/versions/pyenvs/lib/python3.10/site-packages/PIL/Image.py:1043: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
      warnings.warn(
    /home/madan/.pyenv/versions/pyenvs/lib/python3.10/site-packages/PIL/Image.py:1043: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
      warnings.warn(
    /home/madan/.pyenv/versions/pyenvs/lib/python3.10/site-packages/PIL/Image.py:1043: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
      warnings.warn(
    /home/madan/.pyenv/versions/pyenvs/lib/python3.10/site-packages/PIL/Image.py:1043: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
      warnings.warn(
    /home/madan/.pyenv/versions/pyenvs/lib/python3.10/site-packages/PIL/Image.py:1043: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
      warnings.warn(
    /home/madan/.pyenv/versions/pyenvs/lib/python3.10/site-packages/PIL/Image.py:1043: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
      warnings.warn(
    /home/madan/.pyenv/versions/pyenvs/lib/python3.10/site-packages/PIL/Image.py:1043: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
      warnings.warn(
    /home/madan/.pyenv/versions/pyenvs/lib/python3.10/site-packages/PIL/Image.py:1043: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
      warnings.warn(
    /home/madan/.pyenv/versions/pyenvs/lib/python3.10/site-packages/PIL/Image.py:1043: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
      warnings.warn(
    /home/madan/.pyenv/versions/pyenvs/lib/python3.10/site-packages/PIL/Image.py:1043: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
      warnings.warn(



    model.safetensors:   0%|          | 0.00/343M [00:00<?, ?B/s]


    ***** train metrics *****
      epoch                    =         10.0
      total_flos               = 1268772033GF
      train_loss               =       0.2208
      train_runtime            =   0:08:59.74
      train_samples_per_second =       32.571
      train_steps_per_second   =        2.038



```python
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
```



<div>

  <progress value='20' max='20' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [20/20 00:04]
</div>



    /home/madan/.pyenv/versions/pyenvs/lib/python3.10/site-packages/PIL/Image.py:1043: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
      warnings.warn(


    ***** eval metrics *****
      epoch                   =       10.0
      eval_accuracy           =     0.9196
      eval_loss               =     0.3327
      eval_runtime            = 0:00:05.61
      eval_samples_per_second =     55.409
      eval_steps_per_second   =      3.563



```python
kwargs = {
    "finetuned_from": model.config._name_or_path,
    "tasks": "image-classification",
    "dataset": 'house_problem_images',
    "tags": ['image-classification'],
}

if training_args.push_to_hub:
    trainer.push_to_hub('🍻 cheers', **kwargs)
else:
    trainer.create_model_card(**kwargs)
```


```python
# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("MadanKhatri/finetuned-occupations")
model = AutoModelForImageClassification.from_pretrained("MadanKhatri/finetuned-occupations")
```


    preprocessor_config.json:   0%|          | 0.00/353 [00:00<?, ?B/s]


    Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
    /home/madan/.pyenv/versions/pyenvs/lib/python3.10/site-packages/transformers/models/vit/feature_extraction_vit.py:28: FutureWarning: The class ViTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use ViTImageProcessor instead.
      warnings.warn(



    config.json:   0%|          | 0.00/818 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/343M [00:00<?, ?B/s]



```python
from PIL import Image

# Prepare image for the model
image = Image.open("test/test5.png").convert("RGB")  # Open image and convert to RGB
encoding = feature_extractor(image, return_tensors="pt")  # Use your feature_extractor
print(encoding.pixel_values.shape)
```

    torch.Size([1, 3, 224, 224])



```python
import torch

# forward pass
with torch.no_grad():
  outputs = model(**encoding)
  logits = outputs.logits
```


```python
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

    Predicted class: plumber



```python

```
