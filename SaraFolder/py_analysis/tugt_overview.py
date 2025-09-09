from SaraFolder.settings import utils_functions
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    # 406 motion and orientation files
    # In skipped: 132 names
    # Left to analyse: 274
    # df_gt_dict 274
    # all results df is 265 (9 skipped)
    df_fusion = utils_functions.ready_df()

    df_gt = utils_functions.load_groundtruth_samplepersample(df_fusion)
    df_gt_dict = utils_functions.load_groundtruth_dict()

    utils_functions.tugt_overview(df_fusion, df_gt_dict)