import pytest
import numpy as np
from mvlearn.embed import DCCA
from sklearn.exceptions import NotFittedError

@pytest.fixture(scope='module')
def data():
    random_seed = 10
    N = 600
    input_size1, input_size2 = 100, 100
    sin_transform = []
    np.random.seed(random_seed)
    sin_transform.append(20*np.random.rand(N,input_size1))
    sin_transform.append(np.sin(sin_transform[0]))
    layer_sizes1 = [1024, 50]
    layer_sizes2 = [1024, 50]
    dcca_50_2 = DCCA(input_size1, input_size2, n_components=2,
                     layer_sizes1=[1024, 50], layer_sizes2=[1024, 50])
    dcca_print = DCCA(input_size1, input_size2, n_components=2,
                     layer_sizes1=[1024, 50], layer_sizes2=[1024, 50],
                     print_train_log_info=True)
    dcca_low_epochs = DCCA(input_size1, input_size2, n_components=2,
                     layer_sizes1=[1024, 80], layer_sizes2=[1024, 80],
                     epoch_num=10)
    dcca_use_all = DCCA(input_size1, input_size2, n_components=2,
                     layer_sizes1=[1024, 80], layer_sizes2=[1024, 80],
                     use_all_singular_values=True)
    dcca_1layer = DCCA(input_size1, input_size2, n_components=2)

    return {'N' : N, 'input_size1' : input_size1, 'input_size2' : input_size2,
            'layer_sizes1' : layer_sizes1, 'layer_sizes2' : layer_sizes2,
            'sin_transform' : sin_transform, 'random_seed' : random_seed,
            'dcca_50_2' : dcca_50_2, 'dcca_low_epochs' : dcca_low_epochs,
            'dcca_print' : dcca_print, 'dcca_use_all' : dcca_use_all,
            'dcca_1layer': dcca_1layer}

'''
EXCEPTION TESTING
'''

def test_bad_input_size(data):
    # incorrect size
    with pytest.raises(ValueError):
        bad_data1 = []
        bad_data1.append(np.random.rand(50, 10))
        bad_data1.append(np.random.rand(50, data['input_size2']))
        #dcca_50_2.fit(bad_data1)
        data['dcca_50_2'].fit(bad_data1)
    with pytest.raises(ValueError):
        bad_data1 = []
        bad_data1.append(np.random.rand(50, data['input_size1']))
        bad_data1.append(np.random.rand(50, 5))
        # dcca_50_2.fit(bad_data1)
        data['dcca_50_2'].fit(bad_data1)
    with pytest.raises(ValueError):
        dcca = DCCA(0, 5, 2, [10, 10], [10, 10])
    with pytest.raises(ValueError):
        dcca = DCCA(2, 0, 2, [10, 10], [10, 10])
    with pytest.raises(ValueError):
        dcca = DCCA(0.6, 5, 2, [10, 10], [10, 10])
    with pytest.raises(ValueError):
        dcca = DCCA(2, 10.1, 2, [10, 10], [10, 10])

def test_bad_n_components():
    with pytest.raises(ValueError):
        dcca = DCCA(3, 5, 5.5, [10, 10], [10, 10])
    with pytest.raises(ValueError):
        dcca = DCCA(3, 5, 0, [10, 10], [10, 10])
    with pytest.raises(ValueError):
        dcca = DCCA(10, 10, 4, [10, 2], [10, 3])

def test_bad_layer_sizes():
    with pytest.raises(ValueError):
        layer_sizes1, layer_sizes2 = [10, 0], [10, 10]
        dcca = DCCA(3, 5, 2, layer_sizes1, layer_sizes2)
    with pytest.raises(ValueError):
        layer_sizes1, layer_sizes2 = [10, 5.4], [10, 10]
        dcca = DCCA(3, 5, 2, layer_sizes1, layer_sizes2)
    with pytest.raises(ValueError):
        layer_sizes1, layer_sizes2 = [10, 5], [0, 10]
        dcca = DCCA(3, 5, 2, layer_sizes1, layer_sizes2)
    with pytest.raises(ValueError):
        layer_sizes1, layer_sizes2 = [10, 5], [1.9, 10]
        dcca = DCCA(3, 5, 2, layer_sizes1, layer_sizes2)
    with pytest.raises(ValueError):
        layer_sizes1, layer_sizes2 = [10, 5], np.array([10, 10])
        dcca = DCCA(3, 5, 2, layer_sizes1, layer_sizes2)
    with pytest.raises(ValueError):
        layer_sizes1, layer_sizes2 = np.array([10, 10]), [10, 10]
        dcca = DCCA(3, 5, 2, layer_sizes1, layer_sizes2)
    with pytest.raises(ValueError):
        layer_sizes1, layer_sizes2 = (10, 5, 4)
        dcca = DCCA(3, 5, 2, layer_sizes1, layer_sizes2)
    with pytest.raises(ValueError):
        layer_sizes1, layer_sizes2 = [10, 5], [10, 10]
        dcca = DCCA(3, 5, 2, layer_sizes1, layer_sizes2)

def test_bad_epoch_num(data):
    with pytest.raises(ValueError):
        dcca = DCCA(5, 5, 2, data['layer_sizes1'], data['layer_sizes2'],
                    epoch_num=0)
    with pytest.raises(ValueError):
        dcca = DCCA(5, 5, 2, data['layer_sizes1'], data['layer_sizes2'],
                    epoch_num=91.9)

def test_bad_batch_size(data):
    with pytest.raises(ValueError):
        dcca = DCCA(5, 5, 2, data['layer_sizes1'], data['layer_sizes2'],
                    batch_size=0)
    with pytest.raises(ValueError):
        dcca = DCCA(5, 5, 2, data['layer_sizes1'], data['layer_sizes2'],
                    batch_size=91.9)

def test_bad_learning_rate(data):
    with pytest.raises(ValueError):
        dcca = DCCA(5, 5, 2, data['layer_sizes1'], data['layer_sizes2'],
                    learning_rate=0)
    with pytest.raises(ValueError):
        dcca = DCCA(5, 5, 2, data['layer_sizes1'], data['layer_sizes2'],
                    learning_rate=-4)

def test_bad_reg_par(data):
    with pytest.raises(ValueError):
        dcca = DCCA(5, 5, 2, data['layer_sizes1'], data['layer_sizes2'],
                    reg_par=0)
    with pytest.raises(ValueError):
        dcca = DCCA(5, 5, 2, data['layer_sizes1'], data['layer_sizes2'],
                    reg_par=-4)

def test_bad_tolerance(data):
    with pytest.raises(ValueError):
        dcca = DCCA(5, 5, 2, data['layer_sizes1'], data['layer_sizes2'],
                    tolerance=0)
    with pytest.raises(ValueError):
        dcca = DCCA(5, 5, 2, data['layer_sizes1'], data['layer_sizes2'],
                    tolerance=-4)

def test_not_fit_yet(data):
    with pytest.raises(NotFittedError):
        outputs = data['dcca_50_2'].transform(data['sin_transform'])


'''
Performance
'''
def test_sin_transform_performance(data):
    outputs = data['dcca_50_2'].fit_transform(data['sin_transform'])
    corr = np.correlate(outputs[0][:,0], outputs[1][:,0] /
            (outputs[0].shape[0]))
    assert (corr < 1) and (corr > 0)

def test_random_data_fit_transform(data):
    sin_data = data['sin_transform']
    other_data = [np.random.rand(dat.shape[0], dat.shape[1]) for dat in sin_data]
    outputs = data['dcca_1layer'].fit_transform(other_data)
    corr = np.correlate(outputs[0][:,0], outputs[1][:,0] /
            (outputs[0].shape[0]))
    assert (corr < 1) and (corr > 0)

def test_sin_transform_print_train_log(data):
    outputs = data['dcca_print'].fit_transform(data['sin_transform'])

def test_sin_transform_not_converged(data):
    data['dcca_50_2'].fit(data['sin_transform'])
    outputs, loss_good = data['dcca_50_2'].transform(data['sin_transform'],
                                            return_loss=True)
    corr_good = np.correlate(outputs[0][:,0], outputs[1][:,0] /
            (outputs[0].shape[0]))
    data['dcca_low_epochs'].fit_transform(data['sin_transform'])
    outputs, loss_bad = data['dcca_low_epochs'].transform(data['sin_transform'],
                                                  return_loss=True)
    corr_bad = np.correlate(outputs[0][:,0], outputs[1][:,0] /
            (outputs[0].shape[0]))
    assert loss_bad > loss_good
    assert corr_good > corr_bad

def test_sin_transform_use_all_singular_values(data):
    outputs = data['dcca_use_all'].fit_transform(data['sin_transform'])


