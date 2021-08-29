from utils.getter import *
import argparse

parser = argparse.ArgumentParser('Evaluate model on COCO Format')
parser.add_argument('--weight' , type=str, help='checkpoint')
parser.add_argument('--cache_dir', type=str, default=None, help='Cache directory for pre-computed input features')

seed_everything()

def main(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    num_gpus = len(config.gpu_devices.split(','))
    devices_info = get_devices_info(config.gpu_devices)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    trainset, valset, _, valloader = get_dataset_and_dataloader(config)

    if args.cache_dir:
        print("Using cache features")
        trainset.cache_dir = args.cache_dir
        valset.cache_dir = args.cache_dir

    metric = NLPMetrics(valloader, metrics_list=['bleu', "meteor", 'rouge', 'cider', 'spice'])

    net = get_transformer_model(
        patch_size=config.patch_size,
        trg_vocab=trainset.tokenizer.vocab_size)

    net.eval()
    model = Captioning(model = net, device = device)
    model.eval()

    ## Print info
    print(config)
    print(valset)
    print(f"Nubmer of gpus: {num_gpus}")
    print(devices_info)

    if args.weight is not None:                
        load_checkpoint(model, args.weight)
    
    metric.update(model)
    print(metric.value())



if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args.weight)
    if config is None:
        print("Config not found. Load configs from configs/configs.yaml")
        config = Config(os.path.join('configs','configs.yaml'))
    else:
        print("Load configs from weight")   
    main(args, config)