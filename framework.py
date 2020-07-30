import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import os

def get_gpus():
    value = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    value = value.split(',')
    return len(value)

def make_dirs(path:str):
    pos = path.rfind(os.sep)
    if pos < 0:
        raise Exception('Can not find the directory from the path', path)
    path = path[: pos]
    os.makedirs(path, exist_ok=True)

class Config:
    def __init__(self):
        self.lr = 0.001
        self.epoches = 20
        self.batch_size = 10
        self.save_path = '..{sep}models{sep}{name}{sep}{name}'.format(name=self.get_name(), sep=os.sep)
        # self.sample_path = None
        self.logdir = '..{sep}logs{sep}{name}'.format(name=self.get_name(), sep=os.sep)
        self.new_model = False
        self.gpus = get_gpus()

    def get_name(self):
        raise Exception('get_name() is not re-defined.')

    def __repr__(self):
        attrs = self.get_attrs()
        result = ['%s = %s' % (key, attrs[key]) for key in attrs]
        return ', '.join(result)

    def get_attrs(self):
        result = {}
        for attr in dir(self):
            if attr.startswith('__'):
                continue
            value = getattr(self, attr)
            if value is None or type(value) in (int, float, bool, str):
                result[attr] = value
        return result

    def from_cmd(self):
        parser = argparse.ArgumentParser()
        attrs = self.get_attrs()
        for attr in attrs:
            value = attrs[attr]
            t = type(value)
            if t == bool:
                parser.add_argument('--' + attr, default=value, help='Default to %s' % value,
                                    action='store_%s' % ('false' if value else 'true'))
            else:
                parser.add_argument('--' + attr, type=t, default=value, help='Default to %s' % value)
        # parser.add_argument('--call', type=str, default='train', help='Call method, by default call train()')
        a = parser.parse_args()
        for attr in attrs:
            setattr(self, attr, getattr(a, attr))

    # def call(self, name):
    #     if name == 'train':
    #         self.train()
    #     elif name == 'test':
    #         self.test()
    #     else:
    #         print('Unknown method name ' + name, flush=True)

    # def train(self):
    #     app = self.get_app()
    #     with app:
    #         app.train(MyDS(dss.x_train, dss.y_train, cfg), MyDS(dss.x_validation, dss.y_validation, cfg))
    #
    # def test(self):
    #     app = self.get_app()
    #     with app:
    #         test(app, cfg.batch_size, cfg.img_path, cfg.cols)

    def get_tensors(self):
        return Tensors(self)

    def get_sub_tensors(self, gpu_index):
        '''
        Get the sub tensors for the specified gpu.
        :param gpu_index: the index (based on zero) of the GPU
        :return: the sub tensors which has a property 'inputs'
        '''
        raise Exception('The get_sub_tensors() is not defined.')

    def get_app(self):
        return App(self)

    def get_optimizer(self, lr):
        return tf.train.AdamOptimizer(lr)

class Sample:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # print(x_train.shape)
        # print(x_test.shape)
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        x_train, x_validation, y_train, y_validation = \
            train_test_split(x_train, y_train, train_size=55000, test_size=5000)
        # print('x_train shape: ', x_train.shape)
        # print('x_validation shape: ', x_validation.shape)
        # print('y_train shape: ', y_train.shape)
        # print('y_validation shape: ', y_validation.shape)
        self.x_train = x_train
        self.y_train = y_train
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.x_test = x_test
        self.y_test = y_test

class Tensors:
    '''
    provide train_op, summary, sub_ts[i]: {inputs, losses, private_tensors}
    '''
    def __init__(self, config: Config):
        self.config = config
        self.sub_ts = []
        # with tf.variable_scope(config.get_name(), reuse=tf.AUTO_REUSE): #None, True

        with tf.variable_scope(config.get_name()) as scope:
            for i in range(config.gpus):
                with tf.device('/gpu:%d' % i):
                    self.sub_ts.append(config.get_sub_tensors(i))
                    scope.reuse_variables()     #The 1st method to reuse variables
                    # tf.get_variable_scope().reuse_variables()

        with tf.device('/gpu:0'):
            with tf.variable_scope('%s_train' % config.get_name()):

                losses = [ts.losses for ts in self.sub_ts]   #[gpus, losses]
                self.losses = tf.reduce_mean(losses, axis=0)    #[losses]

                self.lr = tf.placeholder(tf.float32, name='lr')
                opt = config.get_optimizer(self.lr)

                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    grads = self.compute_grads(opt)
                    self.apply_grads(grads, opt)

            for i in range(len(losses[0])):
                tf.summary.scalar('loss_%d' % i, self.get_loss_for_summary(self.losses[i]))
            self.summary = tf.summary.merge_all()

    def get_loss_for_summary(self, loss):
        return loss

    def apply_grads(self, grads, opt):
        self.train_ops = [opt.apply_gradients(gs) for gs in grads]

    def compute_grads(self, opt):
        grads = [[opt.compute_gradients(loss) for loss in ts.losses] for ts in self.sub_ts] # [gpus, losses]
        return [self.get_grads_mean(grads, i) for i in range(len(grads[0]))]

    def get_grads_mean(self, grads, loss_idx):
        # grads: [gpus, losses]
        grads = [gs[loss_idx] for gs in grads]     # [gpus]
        vars = [pair[1] for pair in grads[0]]
        result = []
        for i, var in enumerate(vars):
            result.append((tf.reduce_mean([gs[i][0] for gs in grads], axis=0), var))
        return result


class App:
    def __init__(self, config: Config):     # tensors, samples, session, saver
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            self.ts = config.get_tensors()
            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True     # allow some ops to be implemented on CPU if GPU implementation is not an option.
            self.session = tf.Session(config=cfg, graph=graph)
            self.saver = tf.train.Saver()
            if config.new_model:
                self.session.run(tf.global_variables_initializer())
                print('Use a new empty model.')
            else:
                try:
                    self.saver.restore(self.session, config.save_path)
                    print('Restore model from %s successfully' % config.save_path)
                except:
                    print('Fail to restore model from %s, use a new emtpy model instead' % config.save_path)
                    self.session.run(tf.global_variables_initializer())

    def close(self):
        self.session.close()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def train(self, x_train, x_validation):
        self.before_train()
        cfg = self.config
        ts = self.ts
        writer = tf.summary.FileWriter(cfg.logdir, self.session.graph)
        batches = x_train.num_examples // (cfg.batch_size * cfg.gpus)

        for epoch in range(cfg.epoches):
            self.before_epoch(epoch)
            for batch in range(batches):
                self.before_batch(epoch, batch)
                feed_dict = self.get_feed_dict(x_train, batch)
                if len(ts.train_ops) == 1:
                    _, summary = self.session.run([ts.train_ops[0], ts.summary], feed_dict)
                else:
                    for train_op in ts.train_ops:
                        self.session.run([train_op, feed_dict])
                    summary = self.session.run(ts.summary, feed_dict)
                writer.add_summary(summary, epoch * batches + batch)
                self.after_batch(epoch, batch)
                # print('epoch: %d, batch: %03d, loss = %.6f, final_mean[0] = %.3f, final_mean[1] = %.3f, final_mean[2] = %.3f, final_mean[3] = %.3f' % (epoch, batch, loss, final_mean[0], final_mean[1], final_mean[2], final_mean[3]))
            print('Epoch:', epoch, flush=True)
            self.after_epoch(epoch)
        self.after_train()

    def before_train(self):
        print('Training is started', flush=True)

    def before_epoch(self, epoch):
        pass

    def before_batch(self, epoch,batch):
        pass

    def after_train(self):
        print('Training is finished', flush=True)

    def after_epoch(self, epoch):
        self.save()

    def after_batch(self, epoch, batch):
        pass

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('Save model into', self.config.save_path, flush=True)

    def test(self, x_test, y_test):
        pass

    def get_feed_dict(self, xs, batch):
        gpus = self.config.gpus
        cfg = self.config
        result = {self.ts.lr: self.config.lr}
        for i in range(gpus):
            x_values = xs.next_batch(cfg.batch_size, batch)
            for tensor, value in zip(self.ts.sub_ts[i].inputs, x_values):
                result[tensor] = value
        return result
        # y_values = ys[batch * cfg.batch_size: (batch + 1) * cfg.batch_size]
        # return {self.ts.x: x_values, self.ts.y: y_values, self.ts.lr: cfg.lr}
        # return {tensor: value for tensor, value in zip(self.ts.inputs, x_values)}


if __name__ == '__main__':
    cfg = Config()
    cfg.from_cmd()
    print('-' * 100)
    print(cfg)

    dss = Sample()

    app = cfg.get_app()
    with app:
        # app.train(dss.x_train, dss.y_train, dss.x_validation, dss.y_validation)
        app.test(dss.x_test, dss.y_test)