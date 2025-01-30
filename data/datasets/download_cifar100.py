import torchvision 

def download_cifar100(out_path):
    # download the CIFAR10 dataset
    trainset = torchvision.datasets.CIFAR100(
        root=out_path, train=True, download=True)
    testset = torchvision.datasets.CIFAR100(
        root=out_path, train=False, download=True)

out_path = "/fast/slaing/data/vision/cifar100"

if __name__ == "__main__":
    # download the CIFAR10 dataset
    download_cifar100(out_path)