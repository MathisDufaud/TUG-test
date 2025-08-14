from SaraFolder import utils_functions, utils_ML

if __name__ == "__main__":

    df = utils_functions.ready_df()

    utils_ML.ml_pipeline(df)

