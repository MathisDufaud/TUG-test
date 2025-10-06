from SaraFolder.settings import utils_ML, utils_functions, \
    utils_labelling, utils_plots, utils_evaluation
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    # Loading data
    df_fusion = utils_functions.ready_df()
    df_gt = utils_functions.load_groundtruth_samplepersample(df_fusion)
    gt_dict = utils_functions.load_groundtruth_dict()

    # Example with one test:
    if True:
        df_plot = df_fusion[list(df_fusion.keys())[5]]
        res = utils_labelling.full_algo(df_plot)
        utils_plots.plot_labelling_tug(df_plot, res)

    ## Looping all tests:
    all_results = utils_labelling.looping_tests(df_fusion)

    utils_evaluation.evaluate_results(all_results, gt_dict, eval_type='phases')



