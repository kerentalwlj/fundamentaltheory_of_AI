import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import DepthwiseConv2D, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Reshape, Multiply, Permute, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 加载 CIFAR10 数据集
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# 数据预处理
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# ShuffleNet块
def shufflenet_block(x, filters, strides, groups):
    in_channels = x.shape[-1]
    x = Conv2D(filters, kernel_size=1, strides=1, padding='same', groups=groups)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 通道混洗
    x = tf.keras.layers.Reshape((x.shape[1], x.shape[2], groups, filters // groups))(x)
    x = tf.keras.layers.Permute((1, 2, 4, 3))(x)
    x = tf.keras.layers.Reshape((x.shape[1], x.shape[2], filters))(x)

    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x




# 创建 ShuffleNet 模型
input_tensor = Input(shape=(32, 32, 3))

x = Conv2D(24, kernel_size=3, strides=1, padding='same')(input_tensor)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = shufflenet_block(x, 120, 2, 8)
x = shufflenet_block(x, 232, 2, 8)
x = shufflenet_block(x, 464, 2, 8)
x = shufflenet_block(x, 1024, 2, 8)

x = GlobalAveragePooling2D()(x)
output_tensor = Dense(10, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(train_images, train_labels, batch_size=32, epochs=25, validation_data=(test_images, test_labels))

# 绘制训练过程中的准确率和损失
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo-', label='Training acc')
plt.plot(epochs, val_acc, 'r-', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'r-', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
