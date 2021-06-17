import mxnet as mx

from mxnet import gluon
from mxnet import autograd as ag

from network import MLPnet, Convnet

mnist = mx.test_utils.get_mnist()

VCPU = 4
LR = 0.01
EPOCH = 5
BATCH_SIZE = 32 * VCPU

train_data = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], BATCH_SIZE, shuffle=True)
test_data = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], BATCH_SIZE)

gpus = mx.test_utils.list_gpus()
ctx = [mx.cpu(i) for i in range(VCPU)]

net = MLPnet()
# net = Convnet()

net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

acc_metric = mx.metric.Accuracy()
ce_metric = mx.metric.CrossEntropy()

optimizer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': LR})
criterion = gluon.loss.SoftmaxCrossEntropyLoss()

def train():
    train_data.reset()

    for batch in train_data:

        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)

        outputs = []

        with ag.record():
            for x, y in zip(data, label):
                predictions = net(x)

                loss = criterion(predictions, y)
                loss.backward()

                outputs.append(predictions)


        acc_metric.update(label, outputs)
        ce_metric.update(label, outputs)

        optimizer.step(BATCH_SIZE)

    _, acc = acc_metric.get()
    _, loss = ce_metric.get()

    acc_metric.reset()
    ce_metric.reset()

    return acc, loss

def test():
    test_data.reset()

    for batch in test_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)

        outputs = []

        for x in data:
            outputs.append(net(x))

        acc_metric.update(label, outputs)
        ce_metric.update(label, outputs)

        _, acc = acc_metric.get()
        _, loss = ce_metric.get()

        acc_metric.reset()
        ce_metric.reset()

        return acc, loss

for e in range(EPOCH):
    train_acc, train_loss = train()
    test_acc, test_loss = test()

    print(f'Epoch {e}: Accuracy={train_acc:.4f} Loss={train_loss:.4f}\tTest: Accuracy={test_acc:.4f} Loss={test_loss:.4f}')
