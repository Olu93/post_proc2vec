from pathlib import Path

from scipy.stats.stats import RepeatedResults

path_data = Path('data/')
path_figures = Path('figures/')
path_embeddings = path_figures / "embeddings"
path_evaluations = path_figures / "evaluations"
path_models = path_figures / "models"
file_all_datasets = path_data / "all_datasets.pkl"
file_all_datasets_random = path_data / "all_datasets_random.pkl"
file_all_results = path_data / "all_results.csv"
file_all_results_random = path_data / "all_results_random.csv"
file_all_results_cat = path_data / "all_results_categorical.csv"
file_all_results_cat_p2v = path_data / "all_results_categorical_p2v.csv"
file_all_results_cat_a2v = path_data / "all_results_categorical_a2v.csv"
file_all_results_num = path_data / "all_results_numerical.csv"
file_all_results_num_p2v = path_data / "all_results_numerical_p2v.csv"
file_all_results_num_a2v = path_data / "all_results_numerical_a2v.csv"
file_hparams_json = path_data / "hparam_mapper.json"
file_hparams_json_random = path_data / "hparam_mapper_random.json"
file_initial_dataset = path_data / "bpic2012a.csv"
file_embeddings_pdf = path_embeddings / "embeddings.pdf"
file_embeddings_pdf_random = path_embeddings / "embeddings_random.pdf"
file_evaluations_pdf = path_evaluations / "evaluations.pdf"
file_evaluations_pdf_random = path_evaluations / "evaluations_random.pdf"

CV_REPEATS = 10
SEP = "--"
CMB = "_"

column_CaseID = 'Case ID'
column_Activity = 'Activity'
column_Timestamps = 'Complete Timestamp'
column_Remtime = 'remtime'
all_important_cols = [
    column_CaseID,
    column_Activity,
    column_Timestamps,
]
all_time_columns = ["month", "weekday", "hour"]
all_remaining_cols = ["elapsed", "Resource", "AMOUNT_REQ", "open_cases"]

for v in [file_all_datasets, file_all_results, file_hparams_json, file_initial_dataset]:
    print(f"Registered path: {v.absolute()}")