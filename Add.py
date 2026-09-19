from Function import Function

class Add(Function):

    def forward(self,x ,y):
        from Tensor import Tensor   
        z = Tensor(x.data + y.data)
        z.creator = self
        self.inputs = [x,y]
        return z

    def backward(self, grad_output):
        return grad_output, grad_output