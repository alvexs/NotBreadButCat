from v1 import run_demo, recognize, run_training
from v2 import run_demo_v2, recognize_v2, run_training_v2

def demo():
    run_demo()

def predict(target):
    recognize(target)

def train(lr=None, image_dir=None):
    run_training(lr, image_dir)

def demo2():
    run_demo_v2()

def predict2(target):
    recognize_v2(target)

def train2(lr=None, image_dir=None):
    run_training_v2(lr, image_dir)