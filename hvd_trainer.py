import tensorflow as tf
import horovod.tensorflow as hvd

class HorovodTrainer(object):
    """Trainer for neural networks in a distributed environment.

    TensorLayer Trainer is a high-level training interface built on top of TensorFlow MonitoredSession and
    `Horovod <https://github.com/uber/horovod>`__. It transparently scales the training of a TensorLayer model
    from a single GPU to multiple GPUs that be placed on different machines in a single cluster.

    To run the trainer, you will need to install Horovod on your machine. Check the installation script at
    `tensorlayer/scripts/download_and_install_openmpi3_ubuntu.sh`

    The minimal inputs to the Trainer include (1) a training dataset defined using the TensorFlow DataSet API,
    and (2) a model build function given the inputs of the training dataset, and returns the neural network
    to train, the loss function to minimize, and the names of the tensor to log during training, and (3)
    an optimizer and its arguments.

    The default parameter choices of Trainer is inspired by the Facebook paper:
    `Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour <https://arxiv.org/abs/1706.02677>`__

    Parameters
    ----------
    training_dataset : class TensorFlow ``DataSet``
        The training dataset which zips samples and labels. The trainer automatically
        shards the training dataset based on the number of GPUs.
    build_training_func : function
        A function that builds the training operator. It takes the training dataset as an input,
        and returns the neural network, the loss function and a dictionary that maps
        string tags to tensors to log during training.
    optimizer : class TensorFlow ``Optimizer``
        The loss function optimizer. The trainer automatically linearly scale the learning rate based on
        the number of GPUs.
    batch_size : int
        The training mini-batch size (i.e., number of samples per batch).
    prefetch_size: int or None
        The dataset prefetch buffer size. Set this parameter to overlap the GPU training and data preparation
        if the data preparation is heavy.
    checkpoint_dir : None or str
        The path to the TensorFlow model checkpoint. Note that only one trainer master would checkpoints its model.
        If None, checkpoint is disabled.
    log_step_size : int
        The trainer logs training information every N mini-batches (i.e., step size).
    max_iteration: int
        The maximum iteration (i.e., mini-batch) to train.
        The default is `math.inf`. You can set it to a small number to end the training earlier. This is
        usually set for testing purpose.

    Attributes
    ----------
    net : class TensorLayer ``Layer``
        The training model.
    sess : class TensorFlow ``MonitoredTrainingSession``
        The training session tha the Trainer wraps.
    global_step : int
        The number of training mini-batch by far.

    Examples
    --------
    See `tutorial_mnist_distributed_trainer.py
    <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mnist_distributed_trainer.py>`__.

    """

    def __init__(
            self, training_dataset, build_training_func, optimizer, batch_size=32, prefetch_batch=1,
            checkpoint_dir=None, log_step_size=1, max_iteration=float('inf')
    ):
        # Initialize Horovod.
        hvd.init()
        self.is_master = hvd.rank() == 0
        self.parallelism = hvd.size()

        # Get the shard of the dataset based on my local rank
        training_dataset = training_dataset.shard(num_shards=hvd.size(), index=hvd.rank()).batch(batch_size).prefetch(buffer_size=prefetch_batch)
        training_iterator = training_dataset.make_one_shot_iterator()
        self.net, self.loss, self.log_tensors = build_training_func(*training_iterator.get_next())

        # Add Horovod Distributed Optimizer.
        self.opt = hvd.DistributedOptimizer(optimizer)

        self.global_step = tf.train.get_or_create_global_step()
        if isinstance(self.log_tensors, list):
            self.log_tensors.append(self.global_step)
        else:
            self.log_tensors['global_step'] = self.global_step
        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)

        hooks = [
            # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
            # from rank 0 to all other processes. This is necessary to ensure consistent
            # initialization of all workers when training is started with random weights
            # or restored from a checkpoint.
            hvd.BroadcastGlobalVariablesHook(0),

            # Horovod: adjust number of steps based on number of GPUs.
            tf.train.StopAtStepHook(last_step=max_iteration // hvd.size()),
            tf.train.LoggingTensorHook(tensors=self.log_tensors, every_n_iter=log_step_size),
        ]

        # Pin GPU to be used to process local rank (one GPU per process)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())

        # Save checkpoints only on worker 0 to prevent other workers from
        # corrupting them.
        checkpoint_dir = checkpoint_dir if self.is_master else None

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        self.sess = tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir, hooks=hooks, config=config)

