from SaraFolder.settings import utils_labelling, utils_plots, utils_evaluation
from SaraFolder.settings.utils_parkaapp import utils_parkapp
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    # Loading data
    all_tests = utils_parkapp.load_all_tests(dataset_id ='parkapp')

    df_fusion = utils_parkapp.ready_df()
    df_gt = utils_parkapp.load_groundtruth_samplepersample(df_fusion)
    gt_dict = utils_parkapp.load_groundtruth_dict()

    # Example with one test:
    if False:
        df_plot = df_fusion[list(df_fusion.keys())[5]]
        res = utils_labelling.full_algo(df_plot)
        utils_plots.plot_labelling_tug(df_plot, res)

    ## Looping all tests:
    all_tests = utils_labelling.looping_tests(all_tests)

    method = 'labelling'
    title = '_'+method+'resampled'
    dataset = 'parkapp'
    utils_evaluation.evaluate_results(all_tests, eval_type='phases', method=method,
                                      dataset=dataset, title=title)