import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

if __name__ == "__main__":
    ######################################################################### Train
    # Model (MLP)
    # model = Sequential()
    # model.add(Dense(512, input_dim=784, activation="relu")) # 28*28 = 784
    # model.add(Dense(254, activation="relu"))
    # model.add(Dense(10, activation="softmax"))
    # model.summary()

    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # print(X_train.shape) # 60000, 28, 28

    # Convert the shape of data
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    # Normalize
    X_train = X_train.astype("float64") / 255
    X_test = X_test.astype("float64") / 255

    # Label => One-hot-encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Fit
    # model.fit(X_train, y_train, epochs=30, batch_size=200)
    # model.save("./model/mlp_norm_e30_b200.h5") # Save model
    ######################################################################### End of Train

    model = load_model("./model/mlp_e100_b200.h5")
    # acc = model.evaluate(X_test, y_test, batch_size=200)
    # print(acc)
    predict = model.predict(X_test)
    print(predict[0].argmax(), y_test[0].argmax())