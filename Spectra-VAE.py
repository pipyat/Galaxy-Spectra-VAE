from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, History
from tensorflow.keras.optimizers import Adam
from tensorflow import convert_to_tensor
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

#NEURAL NETWORK - VARIATIONAL AUTOENCODER

K.clear_session()

batch_size = 5
original_dim = len(flux[0])
samples = len(flux)
latent_dim = 6
std = epsilon_std =  1
epochs = 20
beta = 0

flux_train = flux[:40000]
flux_test = flux[40000:45000]

x = Input(batch_shape =(None,original_dim,))
encoder_min = Dense(700, activation='elu')(x)
encoder_out = Dense(50, activation='elu')(encoder_in)

#PARAMETERS NEEDED TO SPECIFY MEAN AND STANDARD DEVIATION 
z_mean = Dense(latent_dim)(encoder_out)
z_log_sigma = Dense(latent_dim)(encoder_out)

#SAMPLING FROM DISTRIBUTION - USE REPARAMETERISATION TRICK
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=K.shape(z_mean),
                              mean=0, stddev = epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])


#DECODER
decoder_in =  Dense(50, activation='elu')(z)
decoder_mid = Dense(700, activation='elu')(decoded)
decoder_out = Dense(original_dim, activation='linear')(decoded2)


#Full Model
vae = Model(x, decoded_out)
#Encoder
encoder = Model(x, z_mean)

history = History()
  
# Training stuff
def vae_loss(x, decoded_mean):
      xent_loss = mse(x, decoded_mean)
      kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
      return xent_loss + beta*kl_loss

optimizer = Adam(lr=0.0001)
vae.compile(optimizer=optimizer, loss=vae_loss) 

vae.fit(flux_train,
        flux_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(flux_test,flux_test),
        callbacks = [history])

loss_values = vae.history.history["loss"]
val_loss_values = vae.history.history["val_loss"]
epoch_list = np.arange(1, epochs+1).tolist()

plt.plot(epoch_list, loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.plot(epoch_list, val_loss_values, label='Validation Loss')
plt.legend()

plt.show()

#CLUSTERS
flux_test_encoded2 = encoder.predict(flux_train, batch_size=batch_size)

  
