import torch
from torchreid import models
from collections import OrderedDict


def pytorch_version_to_0_3_1():
    """
    pytorch0.3.1导入0.4.1以上版本模型时加入以下代码块,可对比查看_utils.py文件修正相似错误,
   错误类型为(AttributeError: Can't get attribute '_rebuild_tensor_v2' on
   <module 'torch._utils' from '<pytorch0.3.1>\lib\site-packages\torch\_utils.py'>)
    Returns
    -------
    """
    #  使用以下函数代替torch._utils中的函数(0.3.1中可能不存在或者接口不同导致的报错)
    try:
        torch._utils._rebuild_tensor_v2
    except AttributeError:
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor
        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    try:
        torch._utils._rebuild_parameter
    except AttributeError:
        def _rebuild_parameter(data, requires_grad, backward_hooks):
            param = torch.nn.Parameter(data, requires_grad)
            param._backward_hooks = backward_hooks
            return param
        torch._utils._rebuild_parameter = _rebuild_parameter


def pth_to_jit(model, save_path, device="cuda:0"):
    model.eval()
    input_x = torch.randn(1, 3, 144, 144).to(device)
    new_model = torch.jit.trace(model, input_x)
    torch.jit.save(new_model, save_path)


def jit_to_onnx(jit_model_path, onnx_model_path):
    model = torch.jit.load(jit_model_path, map_location=torch.device('cuda:0'))
    model.eval()
    example_input = torch.randn(1, 3, 144, 144).to("cuda:0")
    example_output = torch.rand(1, 2048, 1, 1).to("cuda:0")
    torch.onnx._export(model, example_input, onnx_model_path, example_outputs=example_output, verbose=True)


def export_model_0_3_1(checkpoint_path, export_model_name, inputsize=[8, 3, 144, 144]):

    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    pytorch_version_to_0_3_1()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = models.init_model(name='resnet50mid', num_classes=6090, loss={'xent', 'htri'}, pretrained=False)
    # model = torch.nn.DataParallel(model).to(device) if device == f"cuda:0" else model
    model = torch.nn.DataParallel(model).cuda() if device == f"cuda:0" else model
    mapped_state_dict = OrderedDict()

    for key, value in checkpoint['state_dict'].items():
        print(key)
        mapped_key = key
        mapped_state_dict[mapped_key] = value
        if 'num_batches_tracked' in key:
            del mapped_state_dict[key]
    model.load_state_dict(mapped_state_dict)
    model.eval()
    model = model.module if device == f"cuda:0" else model
    for key, value in model.state_dict().items():
        print(key)
    dummy_input = torch.autograd.Variable(torch.randn(inputsize))
    dummy_input = dummy_input.cuda() if device == f"cuda:0" else dummy_input
    torch.onnx.export(model=model, args=dummy_input, f=export_model_name, verbose=True)
    # torch.onnx.export(model=model, args=dummy_input, f=export_model_name, verbose=True, input_names=['image'],
    #                   output_names=['outTensor'])


def export_model(checkpoint_path, export_model_name, inputsize=[8, 3, 144, 144]):
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = models.init_model(name='resnet50mid', num_classes=6090, loss={'xent', 'htri'}, pretrained=False)
    # model = torch.nn.DataParallel(model).to(device) if device == f"cuda:0" else model
    model = torch.nn.DataParallel(model).to(device) if device == f"cuda:0" else model
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.module if device == f"cuda:0" else model
    for key, value in model.state_dict().items():
        print(key)
    dummy_input = torch.randn(inputsize).to(device)
    torch.onnx.export(model=model, args=dummy_input, f=export_model_name, verbose=True)
    # torch.onnx.export(model=model, args=dummy_input, f=export_model_name, verbose=True, input_names=['image'],
    #                   output_names=['outTensor'])


def main():
    checkpoint_path = r"./log/resnet50mid-bus_id-xent_htri/model.pth.tar-70"
    export_model_name = checkpoint_path.replace("pth.tar-70", "_70.onnx")
    # export_model(checkpoint_path=checkpoint_path, export_model_name=export_model_name)
    export_model_0_3_1(checkpoint_path=checkpoint_path, export_model_name=export_model_name)


if __name__ == '__main__':
    main()
    # if torch.__version__ < "1.0.0":
    #     print("pytorch version is not  1.0.0, please check it!")
    #     exit(-1)
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # checkpoint_file_path = r"./log/resnet50mid-bus_id-xent_htri/model.pth.tar-90"
    # save_path = r"./log/resnet50mid-bus_id-xent_htri/model_jit.pth.tar-90"
    # checkpoint = torch.load(checkpoint_file_path, map_location=device)
    # model = models.init_model(name="resnet50mid", num_classes=6090, loss={'xent', 'htri'}, pretrained=False)
    # model = torch.nn.DataParallel(model).to(device) if device == f"cuda:0" else model
    # model.load_state_dict(checkpoint['state_dict'])
    # model = model.module
    # model.eval()
    # pth_to_jit(model, save_path, device)
