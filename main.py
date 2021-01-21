import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,models
sns.set()

count=[len(list())]

train_dir=pathlib.Path('C:/Users/Anish/Downloads/archive/images/train')
val_dir=pathlib.Path('C:/Users/Anish/Downloads/archive/images/val')


classes=np.asarray([x.name for x in train_dir.iterdir()])
train_count=[len(list(train_dir.glob(i+'/*.png'))) for i in classes]
val_count=[len(list(val_dir.glob(i+'/*.png'))) for i in classes]

plt.bar(classes,train_count,alpha=0.4)
plt.bar(classes,val_count,alpha=0.4)

plt.legend(['train','val'])

plt.savefig('plots/init_hist.png')


batch_size = 32
img_height = 32
img_width = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
																image_size=(img_height, img_width),
																seed=153)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_dir,
																image_size=(img_height, img_width),
																seed=153)

classes=train_ds.class_names

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])

train_ds = train_ds.repeat(3)



train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

hist_count=[0]*len(classes)
for i,j in train_ds:
    for k in j:
        hist_count[k]+=1

plt.bar(classes,hist_count,alpha=0.4)
plt.savefig('plots/aug_hist.png')


count=1
for i,j in train_ds:
	plt.subplot(6,3,count)
	count+=1
	plt.imshow(i[0])
	plt.axis('off')

plt.savefig('plots/examples.png')


num_classes = len(classes)

model = models.Sequential([
  layers.Conv2D(32, 3, activation='relu', input_shape=(img_height, img_width, 3)),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

fig=plt.figure(figsize=(6,6))


plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.xlabel('Iteration')
plt.ylabel('Accuracy')

plt.legend(['Training','Cross-Validation'])


plt.tight_layout()
plt.savefig('plots/output_measures.png')

