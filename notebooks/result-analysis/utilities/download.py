import os

import pandas as pd

import wandb

api = wandb.Api()


def download_sweep(sweep: str, save_loc: str, override_existing: bool = False):
    if os.path.exists(save_loc):
        if override_existing:
            print(
                f"File {save_loc} already exists; however, overriding with new download."
            )
        else:
            print(
                f"File {save_loc} already exists. Using existing file. Set override_existing=True to overwrite."
            )
            return None
    else:
        print(f"Downloading sweep {sweep} to {save_loc}")

    sweep = api.sweep(sweep)
    runs = sweep.runs

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    # Create a dataframe with summary and config merged into one row
    merged_df = pd.concat(
        [
            pd.DataFrame({"Name": name_list}),
            pd.DataFrame(summary_list),
            pd.DataFrame(config_list),
        ],
        axis=1,
    )

    try:
        merged_df.to_csv(save_loc)
    except:
        print(f"Error saving to {save_loc}")

    return merged_df

    for k, v in run.summary.items():
        if type(v) == float:
            v = round(v, 4)
        print(f"{k}: {v}")

    print(run.config)
    print(merged_df.head())
    print(merged_df.iloc[0].to_dict())


if __name__ == "__main__":
    pass
