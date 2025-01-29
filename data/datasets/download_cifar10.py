import torchvision 

def download_cifar10(out_path):
    # download the CIFAR10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root=out_path, train=True, download=True)
    testset = torchvision.datasets.CIFAR10(
        root=out_path, train=False, download=True)
out_path = "/fast/slaing/data/vision/cifar10"

if __name__ == "__main__":
    # download the CIFAR10 dataset
    download_cifar10(out_path)

