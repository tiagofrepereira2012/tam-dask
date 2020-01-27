import tensorflow as tf
import os
import json



def scale_to_sge(n_workers):
    queue="q_gpu"
    queue_resource_spec="q_gpu=TRUE"
    memory="4GB"
    sge_log= "./logs"
    from dask_jobqueue import SGECluster
    cluster = SGECluster(queue=queue, memory=memory, cores=1, processes=1,
                         log_directory=sge_log,
                         local_directory=sge_log,
                         resource_spec=queue_resource_spec)
    cluster.scale_up(n_workers)
    return Client(cluster)


# Dask using to boostrap the workers
from dask.distributed import Client, LocalCluster
#cluster = LocalCluster(nanny=False, processes=False, n_workers=1, threads_per_worker=1)
#cluster.scale_up(3)
#client = Client(cluster)
from dask_tensorflow import start_tensorflow

#### HERE WE NEED TO WAIT TO GET THE JOBS BEFORE GETTING THE SPEC
client = scale_to_sget(3)
import ipdb; ipdb.set_trace()

###### ONCE YOU GOT THE JOBS, CONTINUE


tf_spec, dask_spec = start_tensorflow(client, ps=1, worker=2)
#os.environ['TF_CONF'] = json.dumps(tf_spec)
os.environ['TF_CONF'] = json.dumps(tf_spec.as_dict())


#tf_spec = {'ps': ['localhost:2222'], 'worker': ['localhost:2223'] }
#os.environ['TF_CONF'] = json.dumps(tf_spec)



def create_mnist_dataset():
    """
    Load MNIST dataset using the tensorflow-datasets (conda install tensorflow-datasets)
    """

    # Loading and scaling
    import tensorflow_datasets as tfds
    dataset = tfds.load(name="mnist", split="train")
    def scale(db):
        image = tf.cast(db['image'], tf.float32) / 255.
        label = db['label']
        return image, label

    dataset = dataset.map(scale)

    dataset = dataset.shuffle(2048).batch(64)

    return dataset
    

class DummyModel(tf.keras.Model):

    def __init__(self):
        from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
        super(DummyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.pool = MaxPool2D(pool_size=(2,2))
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.d1(x)

        return self.d2(x)


#### WRAPPING UP MY CODE INTO THE DISTRIBUTED STRATEGY

#mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = DummyModel()
    dataset = create_mnist_dataset()
    
    # Distributing batched with dataset
    dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
    
    loss_fn =  tf.nn.sparse_softmax_cross_entropy_with_logits

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)


def train_replica_step(inputs, label):
    """
    This step will run in every worker
    """

    with tf.GradientTape() as tape:

        #X = inputs["image"]
        #labels = inputs["label"]

        # MODEL ALREADY INSTANTIATED BEFORE
        logits = model(inputs, training=True)

        # Averaging inside a worker
        loss = tf.nn.compute_average_loss(
                loss_fn(label, logits))

    # Accumulating the gradients
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables))


    return loss


@tf.function
def train_one_epoch(dataset):

    for inputs in dataset:

        l = mirrored_strategy.experimental_run_v2(train_replica_step, 
                                                  args=(inputs[0], inputs[1] ) )

        # Reducing over the workers
        loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, l, axis=0)
        tf.print("Training Loss", loss)
        #tf.print("Training Loss", l)


#### NOW LET'S DO THE OUTER LOOP
epochs = 1
for epoch in range(epochs):
    train_one_epoch(dataset)
pass


