from abc import ABC, abstractmethod


class PosVelFilter():
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def build(self, z=None):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def update(self, z):
        pass

    @abstractmethod
    def save(self):
        pass

    @property
    @abstractmethod
    def x(self):
        pass

    @property
    @abstractmethod
    def dx(self):
        pass
