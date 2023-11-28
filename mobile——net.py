import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import DepthwiseConv2D, BatchNormalization, Activation
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

# MobileNet块
def mobilenet_block(x, filters, strides):
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# 创建 MobileNet 模型
input_tensor = Input(shape=(32, 32, 3))

x = Conv2D(32, kernel_size=3, strides=1, padding='same')(input_tensor)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = mobilenet_block(x, 64, 1)
x = mobilenet_block(x, 128, 2)
x = mobilenet_block(x, 128, 1)
x = mobilenet_block(x, 256, 2)
x = mobilenet_block(x, 256, 1)
x = mobilenet_block(x, 512, 2)
for _ in range(5):
    x = mobilenet_block(x, 512, 1)
x = mobilenet_block(x, 1024, 2)
x = mobilenet_block(x, 1024, 1)

x = GlobalAveragePooling2D()(x)
output_tensor = Dense(10, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

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