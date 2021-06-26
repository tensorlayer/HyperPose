import gdown
import argparse

class Model:
    def __init__(self, id, md5):
        self.id = id
        self.md5 = md5

model_dict = {
    'lopps-resnet50-V2-HW=368x432.onnx': Model('1tb8jnXkoiscfr-ZVydAALg7dtUwAKdEd', 'a6ba26d505c8150d9bf01950143d51d3'),
    'openpose-coco-V2-HW=368x656.onnx': Model('15A0SQyPlU2W-Btcf6Ngi6DY0_1CY50d7', '9f422740c7d41d93d6fe16408b0274ef'),
    'openpose-thin-V2-HW=368x432.onnx': Model('1xqXNFPJgsSjgv-AWdqnobcpRmdIu42eh', '65e26d62fd71dc0047c4c319fa3d9096'),
    'ppn-resnet50-V2-HW=384x384.onnx': Model('1qMSipZ5_QMyRuNQ7ux5isNxwr678ctwG', '0d1df2e61c0f550185d562ec67a5f2ca'),
    'TinyVGG-V1-HW=256x384.uff': Model('1KlKjNMaruJnNYEXQKqzHGqECBAmwB92T', '6551931d16e55cc9370c5c13d91383c3'),
    'openpose-mobile-HW=342x368.onnx': Model('1eDEOC0WBB50bryAbFmhfptyGMoV5wZGn', 'd20e36573a257af450003fd0f17531c1'),
    'openpifpaf-resnet50-HW=368x432.onnx': Model('1cxT1PCPPdMxEdvSB8Q5ewxyTh_TWgcsi', '6c661ded88a91699a1c0582b403d5873'),
    'TinyVGG-V2-HW=342x368.onnx': Model('1ax6fTrxItLXshyHUFTHQVKs5eTRB3t6b', 'c5eb483d18a669c4d61a94233dfb24eb')
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and configure HyperPose models.')
    parser.add_argument('--model', type=str, nargs=1, help='ModelName')

    args = parser.parse_args()
    model_name = args.model[0]

    if model_name not in model_dict.keys():
        print(f'Unknown model resource: {model_name}')
        print('You may use these pretrained models:')
        for k in model_dict.keys():
            print(f'---> {k}')
    else:
        m = model_dict[model_name]
        url = f'https://drive.google.com/uc?id={m.id}'
        gdown.download(url, model_name, quiet=False)
        gdown.cached_download(url, model_name, md5=m.md5)