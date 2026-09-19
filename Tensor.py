from Function import Function
class Tensor():
    def __init__(self,data,requires_grad = False) -> None:
        self.data = data
        self.grad = 0
        self.requires_grad = requires_grad
        self.creator : Function  = None

    def __repr__(self):

        return f"Tensor(data = {self.data}, grad = {self.grad})"

    def __add__(self,other):
        from Add import Add
        result = Add().forward(self,other)
        return result

    def backward(self):
        if self.grad == 0:
            self.grad = 1
        if self.creator is None:
            return

        
        else:
            grads = self.creator.backward(self.grad)
            for i in range(len(self.creator.inputs)):
                self.creator.inputs[i].grad += grads[i]
                self.creator.inputs[i].backward()






x=Tensor(3)
y=Tensor(4)

z=x+y
w=z+y

w.backward()
print(w.grad)
print(x.grad)
print(y.grad)
print(z.grad)