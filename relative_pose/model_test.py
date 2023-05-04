import argparse
from dataset import *
from model.relate_pose_net import RelativePoseNet
import os
from torch.utils.data import DataLoader

# python relative_pose/model_test.py /mnt/gz01/experiment/liuye/relative_pose /mnt/gz01/experiment/liuye/relative_pose
def main():
    parser = argparse.ArgumentParser(description='Read colmap models and output pairs')
    parser.add_argument('sessions_input_folder', help='path to input model folder')
    parser.add_argument('output_folder', help='path to output folder')
    args = parser.parse_args()

    test_dataset = RelativePoseDataset(
        args.sessions_input_folder,
        os.path.join(args.output_folder, "all_data_pairs.pickle"))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    test_item = next(iter(test_dataloader))

    image_1_cuda = test_item["image_1"].cuda()
    image_1_intr_cuda = test_item["param_1"].cuda()
    image_2_cuda = test_item["image_2"].cuda()
    image_2_intr_cuda = test_item["param_2"].cuda()

    logger.info(image_1_cuda.shape)
    logger.info(image_1_intr_cuda.shape)
    logger.info(image_2_cuda.shape)
    logger.info(image_2_intr_cuda.shape)

    model = RelativePoseNet().cuda()
    output = model(image_1_cuda, image_1_intr_cuda, image_2_cuda, image_2_intr_cuda)
    logger.info(output.shape)


    del model
    del output
    del image_1_cuda
    del image_1_intr_cuda
    del image_2_cuda
    del image_2_intr_cuda
    torch.cuda.empty_cache()

    logger.info("Done")


if __name__ == "__main__":
    main()
