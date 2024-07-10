import streamlit as st
import tensorflow as tf
import numpy as np

# Tạo hàm để tạo ảnh ngẫu nhiên
def tao_anh_ngau_nhien():
    image = np.zeros((10*28, 10*28), np.uint8)
    data = np.zeros((100,28,28,1), np.uint8)

    for i in range(0, 100):
        n = np.random.randint(0, 9999)
        sample = st.session_state.X_test[n]
        data[i] = st.session_state.X_test[n]
        x = i // 10
        y = i % 10
        image[x*28:(x+1)*28,y*28:(y+1)*28] = sample[:,:,0]    
    return image, data

# Kiểm tra nếu chưa load model và data, thì load
if 'is_load' not in st.session_state:
    # Load model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1), name='conv2d_input'),
        tf.keras.layers.Conv2D(20, (5, 5), activation='relu', name='conv2d'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='max_pooling2d'),
        tf.keras.layers.Conv2D(50, (5, 5), activation='relu', name='conv2d_1'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1'),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(500, activation='relu', name='dense'),
        tf.keras.layers.Dense(10, activation='softmax', name='dense_1')
    ])

    model.load_weights('digit_weight.h5')

    # Compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    st.session_state.model = model

    # Load data
    (_,_), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_test = X_test.reshape((10000, 28, 28, 1))
    st.session_state.X_test = X_test

    st.session_state.is_load = True
    st.write('Lần đầu load model và data')
else:
    st.write('Đã load model và data rồi')

# Tạo button để tạo ảnh
if st.button('Tạo ảnh'):
    image, data = tao_anh_ngau_nhien()
    st.session_state.image = image
    st.session_state.data = data

# Kiểm tra nếu có ảnh trong session state, thì hiển thị
if 'image' in st.session_state:
    image = st.session_state.image
    st.image(image)

    # Button để nhận dạng
    if st.button('Nhận dạng'):
        data = st.session_state.data
        data = data/255.0
        data = data.astype('float32')
        ket_qua = st.session_state.model.predict(data)
        dem = 0
        s = ''
        for x in ket_qua:
            s = s + '%d ' % (np.argmax(x))
            dem = dem + 1
            if (dem % 10 == 0) and (dem < 100):
                s = s + '\n'    
        st.text(s)
