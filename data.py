import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset
import torch
import pathlib
import torchvision
from torchvision.datasets import CIFAR10, STL10
from torchvision import transforms


def _permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image according to [permutation].

    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order'''

    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]  #--> same permutation for each channel
        image = image.view(c, h, w)
        return image

#ablted flag to allow reading in of original images to all be used as test so no spliting into train and test needed
def get_dataset(name, type='train',ablated=False, download=True, capacity=None, permutation=None, dir='./datasets',
                verbose=False, target_transform=None):
    '''Create [train|valid|test]-dataset.'''

    data_name = 'mnist' if name=='mnist28' else name
    if name not in ('animalpart'):
       dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    dataset_transform = transforms.Compose([
        *AVAILABLE_TRANSFORMS[name],
        transforms.Lambda(lambda x, p=permutation: _permutate_image_pixels(x, p)),
    ])

    # load data-set
    if name =="animalpart" and ablated==True:
        dataset = torchvision.datasets.ImageFolder(f"/content/drive/My Drive/Data/{name}/",transform=dataset_transform, target_transform=target_transform)
    
    elif name in ["animalpart","ablatedhead","ablatedtorso","ablatedtail","allanimalpart"] :
        dataset = torchvision.datasets.ImageFolder(f"/content/drive/My Drive/Data/{name}/",transform=dataset_transform, target_transform=target_transform)
        print(dataset)
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        if type == 'test':
            dataset = test_dataset
        else:
            dataset = train_dataset
i
    else:   
        dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)
        

    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset_copy = copy.deepcopy(dataset)
        dataset = ConcatDataset([dataset_copy for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset


#----------------------------------------------------------------------------------------------------------#


class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


class ExemplarDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, exemplar_sets, target_transform=None):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            total += len(self.exemplar_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            exemplars_in_this_class = len(self.exemplar_sets[class_id])
            if index < (total + exemplars_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
        image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
        return (image, class_id_to_return)


class TransformedDataset(Dataset):
    '''Modify existing dataset with transform; for creating multiple MNIST-permutations w/o loading data every time.'''

    def __init__(self, original_dataset, transform=None, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (input, target) = self.dataset[index]
        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return (input, target)


#----------------------------------------------------------------------------------------------------------#


# specify available data-sets.
AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
    'cifar10': datasets.CIFAR10,
    #'animalpart': pathlib.Path("gdrive/My Drive/Data/animalpart"),
}

# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'mnist28': [
        transforms.ToTensor(),
    ],
    'cifar10': [
        transforms.ToTensor(),
    ],
    'animalpart': [
        transforms.Resize((round(224), round(224))),
        transforms.ToTensor(),
    ],
    'allanimalpart': [
        transforms.Resize((round(224), round(224))),
        transforms.ToTensor(),
    ],
    'ablatedhead': [
        transforms.Resize((round(224), round(224))),
        transforms.ToTensor(),
    ],
    'ablatedtorso': [
        transforms.Resize((round(224), round(224))),
        transforms.ToTensor(),
    ],
    'ablatedtail': [
        transforms.Resize((round(224), round(224))),
        transforms.ToTensor(),
    ],
}

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist28': {'size': 28, 'channels': 1, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'animalpart': {'size': 224, 'channels': 3, 'classes': 8},
    'allanimalpart': {'size': 224, 'channels': 3, 'classes': 8},
    'ablatedhead': {'size': 224, 'channels': 3, 'classes': 8},
    'ablatedtorso': {'size': 224, 'channels': 3, 'classes': 8},
    'ablatedtail': {'size': 224, 'channels': 3, 'classes': 8},
    
}


#----------------------------------------------------------------------------------------------------------#


def get_multitask_experiment(name, scenario, tasks, data_dir="./datasets", only_config=False, verbose=False,
                             exception=False):
    '''Load, organize and return train- and test-dataset for requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''

    # depending on experiment, get and organize the datasets
    if name == 'permMNIST':
        # configurations
        config = DATASET_CONFIGS['mnist']
        classes_per_task = 10
        if not only_config:
            # prepare dataset
            train_dataset = get_dataset('mnist', type="train", permutation=None, dir=data_dir,
                                        target_transform=None, verbose=verbose)
            test_dataset = get_dataset('mnist', type="test", permutation=None, dir=data_dir,
                                       target_transform=None, verbose=verbose)
            # generate permutations
            if exception:
                permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(tasks-1)]
            else:
                permutations = [np.random.permutation(config['size']**2) for _ in range(tasks)]
            # prepare datasets per task
            train_datasets = []
            test_datasets = []
            for task_id, perm in enumerate(permutations):
                target_transform = transforms.Lambda(
                    lambda y, x=task_id: y + x*classes_per_task
                ) if scenario in ('task', 'class') else None
                train_datasets.append(TransformedDataset(
                    train_dataset, transform=transforms.Lambda(lambda x, p=perm: _permutate_image_pixels(x, p)),
                    target_transform=target_transform
                ))
                test_datasets.append(TransformedDataset(
                    test_dataset, transform=transforms.Lambda(lambda x, p=perm: _permutate_image_pixels(x, p)),
                    target_transform=target_transform
                ))
                
    elif name == 'splitMNIST':
        # check for number of tasks
        if tasks>10:
            raise ValueError("Experiment 'splitMNIST' cannot have more than 10 tasks!")
        # configurations
        config = DATASET_CONFIGS['mnist28']
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            # prepare train and test datasets with all classes
            mnist_train = get_dataset('mnist28', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            mnist_test = get_dataset('mnist28', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                train_datasets.append(SubDataset(mnist_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(mnist_test, labels, target_transform=target_transform))
                
    elif name == 'CIFAR10':
        # check for number of tasks
        if tasks>10:
            raise ValueError("Experiment 'CIFAR10' cannot have more than 10 tasks!")
        # configurations
        config = DATASET_CONFIGS['cifar10']
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
            print("Permutation", permutation)
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            # prepare train and test datasets with all classes
            cifar10_train = get_dataset('cifar10', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            cifar10_test = get_dataset('cifar10', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                train_datasets.append(SubDataset(cifar10_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(cifar10_test, labels, target_transform=target_transform)) 
                
    elif name =='ANIMALPART':
        # check for number of tasks
        if tasks>8:
            raise ValueError("Experiment 'ANIMALPART' cannot have more than 8 tasks!")
        # configurations
        config = DATASET_CONFIGS['animalpart']
        classes_per_task = int(np.floor(8 / tasks))
        ##################### REMOVE #################
        print("Class per task", classes_per_task)
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(8))) if exception else np.random.permutation(list(range(8)))
            print("Permutation", permutation)
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            
            # prepare train and test datasets with all classes
            animalpart_train = get_dataset('animalpart', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            animalpart_test = get_dataset('animalpart', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            ##################### REMOVE #################
            print("GENERATED LABELS", labels_per_task)
            
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            if name in ['ABLATEDHEAD','ABLATEDTORSO','ABLATEDTAIL']:
                ablatedtrain_datasets=[]
                ablatedtest_datasets=[]
                
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                train_datasets.append(SubDataset(animalpart_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(animalpart_test, labels, target_transform=target_transform))
                    
               
    elif name in ['ABLATEDHEAD','ABLATEDTORSO','ABLATEDTAIL']:
        # check for number of tasks
        if tasks>8:
            raise ValueError("Experiment 'ANIMALPART' cannot have more than 8 tasks!")
        # configurations
        config = DATASET_CONFIGS['animalpart']
        classes_per_task = int(np.floor(8 / tasks))
        ##################### REMOVE #################
        print("Class per task", classes_per_task)
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(8))) if exception else np.random.permutation(list(range(8)))
            print("Permutation", permutation)
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            
            # prepare original images for test datasets with all classes
            animalpart_test = get_dataset('animalpart', type="test", ablated=True , dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            #for Ablated head
            if name=="ABLATEDHEAD":
                ablated_train = get_dataset('ablatedhead', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
                ablated_test = get_dataset('ablatedhead', type="test", dir=data_dir, target_transform=target_transform,
                                         verbose=verbose)
            #for Ablated torso
            if name=="ABLATEDTORSO":
                ablated_train = get_dataset('ablatedtorso', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
                ablated_test = get_dataset('ablatedtorso', type="test", dir=data_dir, target_transform=target_transform,
                                         verbose=verbose)
                
            #for Ablated tail
            if name=="ABLATEDTAIL":
                ablated_train = get_dataset('ablatedtail', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
                ablated_test = get_dataset('ablatedtail', type="test", dir=data_dir, target_transform=target_transform,
                                         verbose=verbose)
            
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            ##################### REMOVE #################
            print("GENERATED LABELS", labels_per_task)
            
            # split them up into sub-tasks
            test_datasets = []
            ablatedtrain_datasets=[]
            ablatedtest_datasets=[]
                
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                test_datasets.append(SubDataset(animalpart_test, labels, target_transform=target_transform))
                ablatedtrain_datasets.append(SubDataset(ablated_train, labels, target_transform=target_transform))
                ablatedtest_datasets.append(SubDataset(ablated_test, labels, target_transform=target_transform))
                    
    elif name == 'ALLANIMALPART': ############### RECHECK NOT TOO SURE#########!!!!!
        # check for number of tasks
        if tasks>8:
            raise ValueError("Experiment 'ANIMALPART' cannot have more than 3 tasks!")
        # configurations
        config = DATASET_CONFIGS['allanimalpart']
        classes_per_task = 3
        ##################### REMOVE #################
        print("Class per task", classes_per_task)
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(8))) if exception else np.random.permutation(list(range(8)))
            print("Permutation", permutation)
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            
            # prepare train and test datasets with all classes
            animalpart_test = get_dataset('animalpart', type="test",ablated=True, dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            
            ablatedhead_train = get_dataset('ablatedhead', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            ablatedhead_test = get_dataset('ablatedhead', type="test", dir=data_dir, target_transform=target_transform,
                                         verbose=verbose)
     
            ablatedtorso_train = get_dataset('ablatedtorso', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            ablatedtorso_test = get_dataset('ablatedtorso', type="test", dir=data_dir, target_transform=target_transform,
                                         verbose=verbose)
                
            ablatedtail_train = get_dataset('ablatedtail', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            ablatedtail_test = get_dataset('ablatedtail', type="test", dir=data_dir, target_transform=target_transform,
                                         verbose=verbose)
            
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            ##################### REMOVE #################
            print("GENERATED LABELS", labels_per_task)
            
            # split them up into sub-tasks
            test_datasets = []
            ablatedtrain_datasets=[]
            ablatedtest_datasets=[]
                
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                test_datasets.append(SubDataset(animalpart_test, labels, target_transform=target_transform))
                ablatedtrain_datasets.append(SubDataset(ablated_train, labels, target_transform=target_transform))
                ablatedtest_datasets.append(SubDataset(ablated_test, labels, target_transform=target_transform))            
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task if scenario=='domain' else classes_per_task*tasks

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    if name in ['ABLATEDHEAD','ABLATEDTORSO','ABLATEDTAIL']:
        return config if only_config else ((ablatedtrain_datasets,ablatedtest_datasets,test_datasets), config, classes_per_task)
    else:
        return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)
