#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[24]:


from tensorflow.examples.tutorials.mnist import input_data
database=input_data.read_data_sets('/content/data',one_hot=True)


# In[9]:


learning_rate=0.001
num_epochs=30000
batch_size=32


image_dimension=784
nn_dim=512 #encoder dim and decoder dim

latent_variable_dim=2


# In[10]:


def xavier_init(input_shape):
    val=tf.random.normal(shape=input_shape,stddev=1/tf.sqrt(input_shape[0]/2.))
    return val


# In[11]:


weight={'weight_matrix_encoder_hidden':tf.Variable(xavier_init([image_dimension,nn_dim])),
        'weight_mean_hidden':tf.Variable(xavier_init([nn_dim,latent_variable_dim])),
        'weight_std_hidden':tf.Variable(xavier_init([nn_dim,latent_variable_dim])),
        'weight_matrix_decoder_hidden':tf.Variable(xavier_init([latent_variable_dim,nn_dim])),
        'weight_decoder':tf.Variable(xavier_init([nn_dim,image_dimension]))
        }


# In[12]:


bias={'bias_matrix_encoder_hidden':tf.Variable(xavier_init([nn_dim])),
        'bias_mean_hidden':tf.Variable(xavier_init([latent_variable_dim])),
        'bias_std_hidden':tf.Variable(xavier_init([latent_variable_dim])),
        'bias_matrix_decoder_hidden':tf.Variable(xavier_init([nn_dim])),
        'bias_decoder':tf.Variable(xavier_init([image_dimension]))
        }


# In[22]:


image_x=tf.placeholder(tf.float32,shape=[None,image_dimension])#[None,784]any nmbr of rows
encoder_layer=tf.add(tf.matmul(image_x,weight['weight_matrix_encoder_hidden']),bias['bias_matrix_encoder_hidden'])
encoder_layer=tf.nn.tanh(encoder_layer)

mean_layer=tf.add(tf.matmul(encoder_layer,weight['weight_mean_hidden']),bias['bias_mean_hidden'])
std_layer=tf.add(tf.matmul(encoder_layer,weight['weight_std_hidden']),bias['bias_std_hidden'])


# In[14]:


epsilon=tf.random.normal(tf.shape(std_layer),dtype=tf.float32,mean=0.0,stddev=1.0)
latent_layer=mean_layer+tf.exp(0.5*std_layer)*epsilon


# In[15]:


decoder_hidden=tf.add(tf.matmul(latent_layer,weight['weight_matrix_decoder_hidden']),bias['bias_matrix_decoder_hidden'])
decoder_hidden=tf.nn.tanh(decoder_hidden)

decoder_op_layer=tf.add(tf.matmul(decoder_hidden,weight['weight_decoder']),bias['bias_decoder'])
decoder_op_layer=tf.nn.sigmoid(decoder_op_layer)


# In[16]:


def loss_function(original_image,reconstructed_image):
    data_fidelity_loss=original_image*tf.log(1e-10+reconstructed_image)+(1-original_image)*tf.log(1e-10+1-reconstructed_image)
    data_fidelity_loss=-tf.reduce_sum(data_fidelity_loss,1)
    
    kl_div_loss=1+std_layer-tf.square(mean_layer)-tf.exp(std_layer)
    kl_div_loss=-0.5*tf.reduce_sum(kl_div_loss,1)
    
    
    alpha=1
    beta=1
    network_loss=tf.reduce_mean(alpha*data_fidelity_loss+beta*kl_div_loss)
    return network_loss


# In[18]:


loss_value=loss_function(image_x,decoder_op_layer)
optimizer=tf.train.RMSPropOptimizer(learning_rate).minimize(loss_value)


# In[21]:


init=tf.global_variables_initializer()


# In[ ]:


sess=tf.Session()
sess.run(init)

for i in range(num_epochs):
    x_batch,_=database.train.next_batch(batch_size)
    _,loss=sess.run([optimizer,loss_value],feed_dict={image_x:x_batch})
    
    if(i%5000)==0:
        print(f'Iteration {i}/{num_epochs} Loss:{loss}')


# In[23]:


x_noise=tf.placeholder(tf.float32,shape=[None,latent_variable_dim])

decoder_hidden=tf.add(tf.matmul(x_noise,weight['weight_matrix_decoder_hidden']),bias['bias_matrix_decoder_hidden'])
decoder_hidden=tf.nn.tanh(decoder_hidden)

decoder_op_layer=tf.add(tf.matmul(decoder_hidden,weight['weight_decoder']),bias['bias_decoder'])
decoder_op_layer=tf.nn.sigmoid(decoder_op_layer)


# In[ ]:


n=20
x_limit=np.linspace(-2,2,n)
y_limit=np.linspace(-2,2,n)

empty_image=np.empty((28*n,28*n))

for i,zi in enumerate(x_limit):
    for j,pi in enumerate(y_limit):
        generated_latent_layer=np.array([[zi,pi]]*batch_size)
        #generated_latent_layer=np.random.normal(0,1,size=[batch_size,latent_variable_dim])
        generated_image=sess.run(decoder_op_layer,feed_dict={x_noise:generated_latent_layer})
        empty_image[(n-i-1)*28:(n-i)*28,j*28:(j+1)*28]=generated_image[0].reshape(28,28)
plt.figure(figsize=(10,10))
x,y=np.meshgrid(x_limit,y_limit)
plt.imshow(empty_image,origin='upper',cmap='gray')
plt.show()


# In[ ]:


x_sample,y_sample=database.test.next_batch(batch_size+15000)
print(x_sample.shape)

interim=sess.run(latent_layer,feed_dict={image_x:x_sample})
print(interim.shape)

colors=np.argmax(y_sample,i)
plt.figure(figsize=(8,6))
plt.scatter(interim[:,0],interim[:,1],c=colors,cmap='viridis')
plt.colorbar()
plt.grid()
sess.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




