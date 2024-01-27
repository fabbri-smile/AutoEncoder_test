
import tensorflow as tf

from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras import datasets

from dataclasses import dataclass

import matplotlib.pyplot as plt
from PIL import Image

#================================================================================
# ネットワーク定義
#================================================================================
'''
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
'''

@dataclass
class VAELoss:
    rc: float
    kl: float

    @property
    def total(self):
        return self.rc + self.kl

class CAE(Model):

    def __init__(self, optimizer):
        super().__init__()
        self.encoder = Encoder(latent_dim=20)
        self.decoder = Decoder()
        self.loss_function = LossFunction()
        self.optimizer = optimizer

    def call(self, x: tf.Tensor):
        z = self.encoder(x)
        y = self.decoder(z)
        return z, y

    @tf.function
    def train_step(self, y_true: tf.Tensor) -> (tf.Tensor, tf.Tensor, float):
        with tf.GradientTape() as tape:
            z, y = self.call(y_true)
            loss = self.loss_function(y, y_true)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return z, y, loss

    def test_step(self, y_test: tf.Tensor) -> (tf.Tensor, tf.Tensor, float):
        z, y = self.call(y_test)
        loss = self.loss_function(y, y_test)
        return z, y, loss

    def predict(self, image):
        tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        tensor = tf.expand_dims(tensor, axis=0)
        z, y = self.call(tensor)
        z = tf.squeeze(z, [0])
        y = tf.squeeze(y, [0])
        return z, y

class Encoder(layers.Layer):

    def __init__(self, latent_dim=10):
        super().__init__()
        self.c1 = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")
        self.c2 = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")
        self.f1 = layers.Flatten()
        self.d1 = layers.Dense(16, activation="relu")
        self.d2 = layers.Dense(latent_dim, name="z")

    def call(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.f1(x)
        x = self.d1(x)
        z = self.d2(x)
        return z

class Decoder(layers.Layer):

    def __init__(self):
        super().__init__()
        self.d1 = layers.Dense(7 * 7 * 64, activation="relu")
        self.r1 = layers.Reshape((7, 7, 64))
        self.c1 = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.c2 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.c3 = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")

    def call(self, x):
        x = self.d1(x)
        x = self.r1(x)
        x = self.c1(x)
        x = self.c2(x)
        y = self.c3(x)
        return y

class LossFunction:

    def __call__(self, y, y_true):
        rc_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(y_true,  y), axis=(1, 2)
            )
        )
        return rc_loss

#================================================================================
# メトリクスの定義
#================================================================================
'''
import tensorflow as tf
from tensorflow.keras import metrics
'''

class CAETrainMetrics:

    def __init__(self):
        self.loss = metrics.Mean(name='train_rc_loss')

    def update_on_train(self, z, y, loss):
        self.loss.update_state(loss)

    def update_on_test(self, z, y, loss):
        pass

    def reset(self):
        self.loss.reset_states()

    def display(self, epoch: int):
        template = 'Epoch {}, Loss: {:.2g}'
        print(
            template.format(
                epoch,
                self.loss.result()
            )
        )

#================================================================================
# 学習器の定義
#================================================================================

class CAETrainer:

    def __init__(self,  model,  metrics):
        self.model = model
        self.metrics = metrics

    def fit(self, dataset,  epochs: int = 3):
        """ Train mdoel by epochs
        """
        for e in range(1, epochs + 1):

            # train model
            for images in dataset.train_loop():
                z, y, loss = self.model.train_step(images)
                self.metrics.update_on_train(z, y, loss)

            # test model
            for images in dataset.test_loop():
                z, y, loss = self.model.test_step(images)
                self.metrics.update_on_test(z, y, loss)

            # show metrics
            self.metrics.display(e)
            self.metrics.reset()

#================================================================================
# データセットの定義
#================================================================================
'''
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
'''

def build_images_dataset(images, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(
        images
    ).batch(batch_size)
    return dataset

class FashionMnistDataset:
    class_names = (
        'T-shirt/top',
        'Trouser', 
        'Pullover', 
        'Dress', 
        'Coat', 
        'Sandal', 
        'Shirt', 
        'Sneaker',
        'Bag', 
        'Ankle boot')

    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
        self.train_images = train_images.astype("float32") / 255.0
        self.train_labels = train_labels
        self.test_images = test_images.astype("float32") / 255.0
        self.test_labels = test_labels

        self.train_images = self.train_images.reshape((*self.train_images.shape, 1))
        self.test_images = self.test_images.reshape((*self.test_images.shape, 1))

    def info(self):
        return dict(
            train=dict(
                shape=self.train_images.shape,
                size=len(self.train_images)
            ),
            test=dict(
                shape=self.test_images.shape,
                size=len(self.test_images)
            ),
        )

    def get_train_image(self, i: int):
        return self.train_images[i]
    
    def get_test_image(self, i: int):
        return self.test_images[i]

    def get_train_label_name(self, i: int):
        return self.class_names[self.train_labels[i]]

    def display(self):
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.get_train_image(i).squeeze(), cmap=plt.cm.binary)
            plt.xlabel(self.get_train_label_name(i))

    def train_loop(self, batch_size: int = 28):
        return build_images_dataset(self.train_images)

    def test_loop(self):
        return  build_images_dataset(self.test_images)

#================================================================================
# 各クラスの初期化
#================================================================================
'''
from tensorflow.keras import optimizers
'''

# モデル初期化
cae = CAE(optimizer=optimizers.Adam())

# メトリクス初期化
cae_metrics = CAETrainMetrics()

# 学習器初期化
cae_trainer = CAETrainer(cae, cae_metrics)

# データセットクラス初期化
fashion_mnist_dataset = FashionMnistDataset()

#================================================================================
# データの可視化
#================================================================================

fashion_mnist_dataset.info()

fashion_mnist_dataset.display()

#================================================================================
# 学習
#================================================================================

# 学習実行
cae_trainer.fit(fashion_mnist_dataset, epochs=3)

#================================================================================
# 学習結果
#================================================================================
'''
from PIL import Image
'''

class ImageView:

    def display(self, image):
        if hasattr(image, 'numpy'):
            image = image.numpy()
        plt.figure(figsize=(3,3))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image.squeeze(), cmap=plt.cm.binary)

INDEX = 0
image = fashion_mnist_dataset.get_test_image(INDEX)
z, y = cae.predict(image)

view = ImageView()
view.display(image)

# view.display(y)

print('finished')