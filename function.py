from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):

    @abstractmethod
    def init_weight(self, head: int, tail: int) -> float:
        pass

    @abstractmethod
    def value(self, X):
        pass

    @abstractmethod
    def diff(self, Y):
        pass

class Sigmoid(ActivationFunction):
    def init_weight(self, head, tail):
        return np.sqrt(2 / (head + tail))

    def value(self, X):
        # 警告を防ぐため、-250 から 250 の範囲にクリッピング（速度アップと安定化）
        X_clipped = np.clip(X, -250, 250)
        exp = np.exp(-X_clipped)
        return 1 / (exp + 1)
    
    def diff(self, Y):
        return Y - Y**2
    
class Tanh(ActivationFunction):
    def init_weight(self, head, tail):
        return np.sqrt(2 / (head + tail))

    def value(self, X):
        return np.tanh(X)
    
    def diff(self, Y):
        return 1 - Y**2
    
class ReLU(ActivationFunction):
    def init_weight(self, head, tail):
        return np.sqrt(2 / head)
    
    def value(self, X):
        eps = 1e-15
        return np.maximum(-eps, X)
    
    def diff(self, Y):
        return (Y >= 0).astype(float)
    
class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def init_weight(self, head, tail):
        return np.sqrt(2 / head)
    
    def value(self, X):
        return np.maximum(X, X * self.alpha)
    
    def diff(self, Y):
        # Yが0より大きければ1.0、小さければalphaの配列を作る
        # np.ones_like で Y と同じ形の 1.0 の配列を作り、0以下の場所を alpha で上書き
        d = np.ones_like(Y)
        d[Y < 0] = self.alpha
        return d



class OutputFunction(ABC):
    @abstractmethod
    def value(self, X):
        pass

    @abstractmethod
    def Loss(self, P, Y):
        pass

    @abstractmethod
    def dLoss(self, P, Y):
        pass

class Softmax(OutputFunction):        
    def value(self, X):
        eps = 1e-15
        X_max = np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X - X_max)
        sum_exp = np.sum(exp_X,axis=1,keepdims=True)
        return exp_X / (sum_exp + eps)
    
    def Loss(self, P, Y):
        # Pが0や1にならないように極小値を挟む        
        eps = 1e-15
        P_clipped = np.clip(P, eps, 1 - eps)
        logP = np.log(P_clipped)
        batch_size = P.shape[0]
        loss = -np.sum(Y * logP) / batch_size
        return loss
   
    def dLoss(self, P, Y):
        batch_size = P.shape[0]
        return (P - Y) / batch_size
    
class Identity(OutputFunction):   
    def value(self, X):
        return X
    
    def Loss(self, P, Y):
        return np.mean((P - Y)**2)
    
    def dLoss(self, P, Y):
        size_P = P.size
        return 2*(P - Y) / size_P