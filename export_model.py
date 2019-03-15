import torch
from torchreid import models


def export_model(checkpoint_path, export_model_name, inputsize=[1, 3, 144, 144]):
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = models.init_model(name='mlfn', num_classes=10905, loss={'xent', 'htri'}, pretrained=False).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    for key, value in model.state_dict().items():
        print(key)
    # features = model.model.model.features
    # global_avg_pool = torch.nn.AvgPool2d(5, 5)
    # model = torch.nn.Sequential(OrderedDict([
    #     ("features", features),
    #     ("global_avgpool", global_avg_pool)]))
    dummy_input = torch.randn(inputsize).to(device)
    torch.onnx.export(model=model, args=dummy_input, f=export_model_name, verbose=True)
    # torch.onnx.export(model=model, args=dummy_input, f=export_model_name, verbose=True, input_names=['image'],
    #                   output_names=['outTensor'])


def main():
    checkpoint_path = r"./log/mlfn-bus_id-xent_htri/checkpoint_ep170.pth.tar"
    export_model_name = checkpoint_path.replace("pth.tar", "onnx")
    export_model(checkpoint_path=checkpoint_path, export_model_name=export_model_name)


if __name__ == '__main__':
    main()
