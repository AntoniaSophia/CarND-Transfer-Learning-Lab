import zipfile
import pickle

def import_trafficsign(archive_file):
    archive = zipfile.ZipFile(archive_file, 'r')

    training_file = 'vgg_cifar10_100_bottleneck_features_train.p'
    validation_file= 'vgg_cifar10_bottleneck_features_validation.p'

    with archive.open(training_file, mode='r') as f:
        train = pickle.load(f)
    with archive.open(validation_file, mode='r') as f:
        valid = pickle.load(f)
    print(train)

    result = {
        'X_train': train['features'],
        'y_train': train['labels'],
        'X_valid': valid['features'],
        'y_valid': valid['labels']
    }

    return result