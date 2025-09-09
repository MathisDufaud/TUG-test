from SaraFolder.settings import utils_ML, utils_functions, utils_labelling, utils_plots

if __name__ == "__main__":
    # Loading data
    df_fusion = utils_functions.ready_df()
    choice = '1_1_35254651_s'
    df_plot = df_fusion[choice]
    res = utils_labelling.full_algo(df_plot)

    utils_plots.plot_labelling_tug(df_plot, res)

    ## Looping all tests:
    all_results = utils_labelling.looping_tests(df_fusion)



