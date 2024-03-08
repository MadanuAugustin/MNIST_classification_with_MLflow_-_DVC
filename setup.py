




import setuptools

from setuptools import find_packages, setup


with open("README.md", 'r', encoding='utf-8') as f:
    long_description = f.read()




REPO_NAME = 'MNIST-classification-with-Mlflow & DVC'
AUTHOR_USER_NAME = 'augustin'
SRC_REPO = 'MNIST classifier'
AUTHOR_EMAIL = 'augustin7766@gmail.com'


setup(
    name = SRC_REPO,
    version= '0.0.0.0',
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description='python application for MNIST classification',
    long_description=long_description,
    url=f"https://github.com/MadanuAugustin/MNIST_classification_with_MLflow_and_DVC.git",
    project_urls = {
        "Bug Tracker" : f"https://github.com/MadanuAugustin/MNIST_classification_with_MLflow_and_DVC.git"
    },
    package_dir={"":"src"},
    packages=find_packages(where="src")
)
