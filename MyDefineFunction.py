#-*-coding:utf-8-*-
import torch
from torch.autograd import Variable
from torch.autograd import Function

# 定以输入参数是Variable的Function
class LinearFunction(Function):
    # 创建torch.autograd.Function类的一个子类
    # 必须是staticmethod
    @staticmethod
    # 第一个是ctx，第二个是input，其他是可选参数。
    # ctx在这里类似self，ctx的属性可以在backward中调用。
    # 自己定义的Function中的forward()方法，所有的Variable参数将会转成tensor！因此这里的input也是tensor．
    # 在传入forward前，autograd engine会自动将Variable unpack成Tensor。
    def forward(ctx, input, weight, bias=None):
        print("type of input is: ",type(input))
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output为反向传播上一级计算得到的梯度值
        input, weight, bias = ctx.saved_variables
        # 分别代表输入,权值,偏置三者的梯度
        print("type of backward is: ") # ('type of input is: ', <class 'torch.DoubleTensor'>)
        # type of backward is: 
        # (<class 'torch.autograd.variable.Variable'>, <class 'torch.autograd.variable.Variable'>, <type 'NoneType'>)
        print(type(input), type(weight), type(bias))
        grad_input = grad_weight = grad_bias = None
        # 判断三者对应的Variable是否需要进行反向求导计算梯度
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)  # 复合函数求导，链式法则
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)  # 复合函数求导，链式法则
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

# 测试自己定义的Function是否正确(看前向计算和反向求导是否正确)
from torch.autograd import gradcheck
linear = LinearFunction.apply
input = (Variable(torch.randn(20,20).double(), requires_grad=True), Variable(torch.randn(30,20).double(), requires_grad=True),)
test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
print(test)  #　没问题的话输出True


#　使用自己定义的LinearFunction来创建nn.Module的子类
class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        # nn.Parameter is a special kind of Variable, that will get
        # automatically registered as Module's parameter once it's assigned
        # 这个很重要！ Parameters是默认需要梯度的！
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)
        # 或者　return LinearFunction()(input, self.weight, self.bias)



# 定义输入参数是Tensor的Function
class MulConstant(Function):
    from torch.autograd.function import once_differentiable
    @staticmethod
    def forward(ctx, tensor, constant):
        # ctx is a context object that can be used to stash information
        # for backward computation
        print("type of input is: ",type(tensor))
        ctx.constant = constant
        return tensor * constant

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        print("type of grad_output is: ",type(grad_output))
        return grad_output * ctx.constant, None  # 这里并没有涉及到Variable

mulcon = MulConstant.apply
input = (torch.randn(20,20).double(), 5) # 传入tensor和constant
test = gradcheck(mulcon, input, eps=1e-6, atol=1e-4)
print(test)  #　没问题的话输出True