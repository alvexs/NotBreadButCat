from v1 import run_demo, recognize, run_training

def demo():
    run_demo()

def predict(target):
    recognize(target)

def train(lr=None, image_dir=None):
    run_training(lr, image_dir)
