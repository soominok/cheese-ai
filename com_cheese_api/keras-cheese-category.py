import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import matplotlib.pyplot as plt


def model_cheese():
    cheese = pd.read_csv("com_cheese_api/resources/data/cheese_dataset.csv")
    cheese_matching = pd.read_csv("com_cheese_api/resources/data/cheese_matching.csv", encoding="utf-8")

    cheese_df = cheese.rename(columns = {"Unnamed: 0": "cheese_no"})
    cheese_matching_df = cheese_matching.rename(columns={"Unnamed: 0": "cheese_no"})

    cheese_data = pd.merge(cheese_df, cheese_matching_df, on="cheese_no", how = "inner")

    cheese_model_data = cheese_data.drop(['Unnamed: 0.1', 'content', 'img', 'brand', 'country', 'matching'], axis = 1)
    cheese_2 = cheese_data.drop(['Unnamed: 0.1', 'content', 'img', 'matching'], axis = 1)
    # cheese_model_data.to_csv("resources/data/cheese_model_data.csv", encoding = 'utf-8-sig')

    # print('cheese_model_data: ', {cheese_model_data})
    # print()

    # X_cheese = cheese_model_data.iloc[:, 3:68]
    # X_cheese = X_cheese.drop(['name', 'category', 'texture', 'types', 'price', 'brand_code', 'country_code'], axis = 1)
    X_cheese = cheese_model_data.drop(['cheese_no', 'ranking', 'cheese_id', 'name', 'category', 'texture', 'types', 'price', 'brand_code', 'country_code'], axis = 1)
    # X_cheese = X_cheese.drop(['name', 'category', 'types', 'price'], axis = 1)
    y_cheese = cheese_model_data[['category']]
    # y_cheese = cheese_model_data[['name']]

    print(X_cheese)

    X_cheese_train, X_cheese_test, y_cheese_train, y_cheese_test = train_test_split(X_cheese, y_cheese, test_size = 0.3)

    X_cheese_train = X_cheese_train.astype('float32')
    X_cheese_test = X_cheese_test.astype('float32')
    print(X_cheese_train.shape)
    model = tf.keras.models.Sequential([       # valid (default 값)
        tf.keras.layers.Dense(128, activation='relu', input_shape=(59,)),
        # tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(20, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

    history = model.fit(X_cheese_train, y_cheese_train, batch_size=10, validation_data=(X_cheese_test, y_cheese_test), epochs=300)

    loss, acc = model.evaluate(X_cheese_test, y_cheese_test, batch_size=10)

    print('loss: ', loss)
    print('acc: ', acc)

    # X_new = np.array([[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    X_new1 = np.array([[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    predictions = model.predict(X_new1)
    # print(predictions)

    output = np.argmax(predictions[0])
    print(output)

    # test_labels[0]

    ###################################################



    history_dict = history.history
    history_dict.keys()


    # 손실도 그래프
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # # "bo"는 "파란색 점"입니다
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # # b는 "파란 실선"입니다
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')

    # "bo"는 "파란색 점"입니다
    plt.plot(epochs, loss, label='Training loss')
    # b는 "파란 실선"입니다
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    # 정확도 그래프
    plt.clf()   # 그림을 초기화합니다

    # plt.plot(epochs, acc, 'bo', label='Training acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.plot(epochs, acc, label='Training acc')
    plt.plot(epochs, val_acc, label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    
    return model

if __name__ == '__main__':
    model = model_cheese()
    model.save("cheese_model.h5")
