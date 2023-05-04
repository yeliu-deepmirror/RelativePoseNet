import argparse
from dataset import *
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# python relative_pose/dataset_test.py /mnt/gz01/experiment/liuye/relative_pose /mnt/gz01/experiment/liuye/relative_pose
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
    logger.info(test_item["param_1"])
    logger.info(test_item["param_2"])
    logger.info(test_item["target_pose"])

    image_1 = test_item["image_1"][0].squeeze()
    image_2 = test_item["image_2"][0].squeeze()

    fig = plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(image_1, cmap="gray")
    plt.subplot(122)
    plt.imshow(image_2, cmap="gray")
    plt.show()

    logger.info("Done")


if __name__ == "__main__":
    main()
