import zipfile
import pickle

def import_trafficsign(archive_file):
    archive = zipfile.ZipFile(archive_file, 'r')

    training_file = 'train.p'
    validation_file= 'valid.p'
    testing_file = 'test.p'

    with archive.open(training_file, mode='r') as f:
        train = pickle.load(f)
    with archive.open(validation_file, mode='r') as f:
        valid = pickle.load(f)
    with archive.open(testing_file, mode='r') as f:
        test = pickle.load(f)
        
    result = {
        'X_train': train['features'],
        'y_train': train['labels'],
        'X_valid': valid['features'],
        'y_valid': valid['labels'],
        'X_test': test['features'],
        'y_test': test['labels']
    }

    return result