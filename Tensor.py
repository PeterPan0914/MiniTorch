class Tensor():
    def __init__(self,data,requires_grad = False) -> None:
        self.data = data
        self.grad = 0
        self.requires_grad = requires_grad
        self.parents = []
        self.creator = ''

    def __repr__(self):

        return f"Tensor(data = {self.data}, grad = {self.grad})"

    def __add__(self,other):
        result = Tensor(self.data + other.data)
        result.parents = [self,other]
        result.creator="add"

        return result


    def backward(self):
        if self.grad == 0:
           self.grad = 1

        if self.parents == []:
            return self.grad
        else:
            for parent in self.parents:
                parent.grad += self.grad
                parent.backward()





x = Tensor(3, requires_grad=True)

y = Tensor(4, requires_grad=True)

z = x+y

z.backward()

print(x.grad)
print(y.grad)