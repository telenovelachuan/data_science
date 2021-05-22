
import pandas as pd
import numpy as np
from numpy import inf
import h5py

# reads all raw data files and stack into two numpy arrays:
# 1. a data array that contains the Euclidean norm, phase and SNR of every signal
# 2. a position array that contains all the ground truth GSP locations


# load data from file_1 - file_9.hdf4
result = None
all_pos = None
CTW_labelled = "/home/c693s270/"

for i in range(9):
    idx = i + 1
    print(f"handling file_{idx}.hdf5...")
    data_file = CTW_labelled + "file_" + str(idx) + ".hdf5"
    H_Re, H_Im, SNR, Pos = get_data(data_file)
    n = H_Re.shape[0]
    print(f"shape[0:{n}")

    # calculating Euclidean norm
    H = np.sqrt(H_Re[:,:]**2 + H_Im[:,:]**2)

    # Extracting phase
    print("extracting phase...")
    arctan = np.divide(H_Re[:,:], H_Im[:,:])
    arctan[np.isnan(arctan)] = math.pi / 2
    #arctan.shape

    # append phase onto H
    print("appending phase...")
    h1 = np.empty([n, 56, 924, 10])
    for d1 in range(n):
        for d2 in range(56):
            for d3 in range(924):
                h = H[d1][d2][d3]
                phase = arctan[d1][d2][d3]
                h1[d1][d2][d3] = np.vstack((h, phase)).reshape(-1)
        

    # append SNR onto h1
    print("appending SNR...")
    snr_rep = np.repeat(SNR, 2, axis=1).reshape(n, 56, -1)
    h2 = np.empty([n, 56, 925, 10])
    for d1 in range(n):
        for d2 in range(56):
            h = h1[d1][d2]
            snr = snr_rep[d1][d2]
            h2[d1][d2] = np.vstack((h, snr))
        

    # replace inifinity values
    h2[h2 == inf] = 0
    h2[h2 == -inf] = 0

    if result is None:
        result = h2
        all_pos = Pos
    else:
        result = np.vstack((result, h2))
        all_pos = np.vstack((all_pos, Pos))

# save to disk
with open('h2-pos.npy', 'wb') as f:
    np.save(f, result)
    np.save(f, all_pos)

# Dimensionality reduction using autoencoder
def data_gen(data):
    for i in range(len(data)):
        yield (data[i: i + 1], data[i: i + 1])

# define the autoencoder structure
def autoencoder(input_shape, opt=Adam(1e-3), dropout_rate=0.2):
    model = Sequential()
    model.add(Conv1D(16, 16, input_shape=input_shape, activation='relu'))
    model.add(Conv1D(32, 16, activation='relu'))
    model.add(Conv1D(32, 16, activation='relu'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3))
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model

ae = autoencoder()
# load numpy preprocessed data
with open('h2-pos.npy', 'rb') as f:
    h2 = np.load(f)
    Pos = np.load(f)

# Train-test split for autoencoder. Labels are training set itself
data, data_v = train_test_split(h2, test_size=0.5, random_state=54) 

# Train autoencoder
for i in range(5):
    ae.fit_generator(data_gen(data),validation_data=data_gen(data_v), epochs=1, steps_per_epoch=len(data),
                     validation_steps=len(data_v))

# Extract the first half of the model as encoder
encoder = Model(ae.input, ae.layers[-5].output)

# Get the dimension-reduced dataset using the encoder
h2_ae = encoder.predict(h2)

# save to disk
with open('all_ae.npy', 'wb') as f:
    np.save(f, h2_ae)
    np.save(f, Pos)

