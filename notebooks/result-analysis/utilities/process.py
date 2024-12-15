from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_results_df(
    df, target_cols, unique_group_channels, unique_groups, verbose=False
):
    """
    Create a pandas table with the results.
    Every row has 3 columns: [col, EP(%), FP(%)].
    """
    results_df = pd.DataFrame(columns=["col", "EP(%)", "FP(%)", "p(l(FP) < l(EP))"])

    for col in target_cols:
        ape, lower_bound, upper_bound, ep_bootstrap = _bootstrap_wrapper(
            df,
            col,
            unique_group_channels,
            unique_groups,
            use_lit_comp=False,
            verbose=verbose,
        )
        ape1_str = f"{ape:.2f} ({lower_bound:.2f}, {upper_bound:.2f})"

        ape, lower_bound, upper_bound, fp_bootstrap = _bootstrap_wrapper(
            df,
            col,
            unique_group_channels,
            unique_groups,
            use_lit_comp=True,
            verbose=verbose,
        )
        ape2_str = f"{ape:.2f} ({lower_bound:.2f}, {upper_bound:.2f})"

        # simulate p(l(FP) < l(EP)); could probably calculate using math.
        N = 1000
        ps = np.empty(N)
        rng = np.random.default_rng()
        for n in range(N):
            ep_samples = rng.permutation(ep_bootstrap, axis=1)
            fp_samples = rng.permutation(fp_bootstrap, axis=1)
            ps[n] = (np.abs(fp_samples) < np.abs(ep_samples)).mean()
        p = ps.mean()

        # calculate p(l(FP) < p(EP))

        # use pandas concat instead of append
        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    {
                        "col": col,
                        "EP(%)": ape1_str,
                        "FP(%)": ape2_str,
                        "p(l(FP) < l(EP))": np.round(p * 100, 2),
                    },
                    index=[0],
                ),
            ]
        )

    return results_df


def _bootstrap_wrapper(
    df,
    col: str,
    unique_group_channels: List,
    unique_groups: List,
    use_lit_comp: bool = False,
    verbose=False,
):
    if verbose:
        print(f"{col}: ---------------")
        print(f"target_pop   \t<-\tpred_pop:\t APE (95% CI)")
        if use_lit_comp:
            print(f"Literature: ---------------")
        else:
            print(f"My method: ---------------")

    if use_lit_comp:
        channel_comparison = zip(unique_group_channels[:-1], unique_group_channels[1:])
    else:
        channel_comparison = zip(unique_group_channels[:-1], unique_group_channels[:-1])

    apes = []
    bootstrapped_apes = []
    for target_group_channels, pred_group_channels in channel_comparison:
        for target_group, pred_group in zip(unique_groups[:-1], unique_groups[1:]):
            target_subset = df[
                (df["group"] == target_group)
                & (df["hidden_group_channels"] == target_group_channels)
            ]
            pred_subset = df[
                (df["group"] == pred_group)
                & (df["hidden_group_channels"] == pred_group_channels)
            ]

            # TODO: temp fix; new sweep shouldn't require
            if len(pred_subset) == 0 or len(target_subset) == 0:
                continue

            target_population = target_subset[col].to_numpy()
            pred_population = pred_subset[col].to_numpy()

            (
                ape,
                lower_bound,
                upper_bound,
                bootstrapped_ape,
            ) = boostrap_population_absolute_percentage_error(
                target_population, pred_population
            )

            apes.append(ape)
            bootstrapped_apes.append(bootstrapped_ape)

            if verbose:
                print(
                    f"{target_group} {target_group_channels}\t <-\t {pred_group} {pred_group_channels}:  \t {ape:.2f} ({lower_bound:.2f}, {upper_bound:.2f})"
                )

    apes = np.array(apes)
    bootstrapped_apes = np.array(bootstrapped_apes)
    lower_bound = np.percentile(bootstrapped_apes, 2.5, axis=0)
    upper_bound = np.percentile(bootstrapped_apes, 97.5, axis=0)

    mean_ape = np.mean(apes)
    lower_bound = np.mean(lower_bound)
    upper_bound = np.mean(upper_bound)

    method = "free parameters" if use_lit_comp else "expanded parameters"
    print(
        f"{col} {method} Overall APE: {mean_ape:.2f} ({lower_bound:.2f}, {upper_bound:.2f})"
    )

    return mean_ape, lower_bound, upper_bound, bootstrapped_apes


def ape_fn(target, pred):
    # absolute percentage error for a target-pred pair
    return (pred - target) / target * 100


def boostrap_population_absolute_percentage_error(
    target_population: np.array,
    pred_population: np.array,
    bootstrap_samples: int = 5000,
):
    """
    Given a target and prediction population, how well does the pred population's mean predict the target population's mean?
    Error metric is absolute percentage error.
    """
    mean_target = np.mean(target_population)
    mean_pred = np.mean(pred_population)
    ape = ape_fn(mean_target, mean_pred)

    # also compute a confidence interval for the APE, using bootstrapping
    bootstrapped_ape = np.empty(bootstrap_samples, dtype=np.float32)
    for i in range(bootstrap_samples):
        # sample indices
        target_indices = np.random.choice(
            len(target_population), len(target_population)
        )
        pred_indices = np.random.choice(len(pred_population), len(pred_population))

        # sample populations
        bootstrap_target = target_population[target_indices]
        bootstrap_pred = pred_population[pred_indices]

        # compute means
        bootstrap_target_mean = np.mean(bootstrap_target)
        bootstrap_pred_mean = np.mean(bootstrap_pred)

        # compute APE
        bootstrapped_ape[i] = ape_fn(bootstrap_target_mean, bootstrap_pred_mean)

    # construct a 95% confidence interval for ape
    bootstrapped_ape.sort()
    lower_bound = bootstrapped_ape[int(0.025 * bootstrap_samples)]
    upper_bound = bootstrapped_ape[int(0.975 * bootstrap_samples)]

    return np.mean(ape), lower_bound, upper_bound, bootstrapped_ape


def mean_coefficient_of_variation(df, col: str):
    # only use all of df for hidden_group_channels comparison
    df_hgc = df

    print(f"{col}: ---------------")

    # group by hidden_group_channels, take the std for every setting
    grouped_col = df_hgc.groupby("hidden_group_channels")[col]
    # calculate mean percentage error of every group
    cv_hgc = grouped_col.std() / grouped_col.mean()
    print(cv_hgc)
    mean_cv_hgc = cv_hgc.mean()

    # only use df where original_literature_comparison_value is in [24, 34]
    df_olcv = df[
        df["original_literature_comparison_value"].isin([8, 12, 16, 24, 34, 48, 68, 96])
    ]

    # group by original_literature_comparison_value, take the std for every setting
    grouped_col = df_olcv.groupby("original_literature_comparison_value")[col]
    cv_olcv = grouped_col.std() / grouped_col.mean()
    print(cv_olcv)
    mean_cv_olcv = cv_olcv.mean()

    return mean_cv_hgc, mean_cv_olcv


# using this example https://matplotlib.org/2.0.2/examples/statistics/violinplot_demo.html
def create_error_plots(
    df: pd.DataFrame,
    col: str,
    unique_groups: List[str],
    unique_group_channels: List[int],
    aggregate_mode="mean",
    bootstrap: bool = False,
):
    """
    Creates confidence intervals for mean at different hidden_group_channels.
    """
    fig, axs = plt.subplots(1, 1)
    fig.suptitle(f"{col} {aggregate_mode} 95% conf interval")

    # for ever group in unique_groups, plot the mean of col as a function of hidden_group_channels
    factor = np.max(df["hidden_group_channels"])
    eps = factor / 85 * np.linspace(-1, 1, len(unique_groups) + 1)

    assert aggregate_mode in ["mean", "median"]
    agg_fn = np.mean if aggregate_mode == "mean" else np.median

    def plot(df, n: int, group: str = "all"):
        if group == "all":
            plot_df = df
            plt_label = "all"
        else:
            plot_df = df[df["group"] == group]
            plt_label = group

        def lower_percentile(vals):
            if bootstrap:
                B = 1000
                bootstrap_means = np.empty(B, dtype=np.float32)
                for b in range(B):
                    bootstrap_vals = np.random.choice(
                        vals, size=len(vals), replace=True
                    )
                    bootstrap_means[b] = agg_fn(bootstrap_vals)
                return np.percentile(bootstrap_means, 2.5, axis=0)
            else:
                std_err = np.std(vals) / np.sqrt(len(vals) - 1)
                lower = agg_fn(vals) - 1.96 * std_err
                return lower

        def upper_percentile(vals):
            if bootstrap:
                B = 1000
                bootstrap_means = np.empty(B, dtype=np.float32)
                for b in range(B):
                    bootstrap_vals = np.random.choice(
                        vals, size=len(vals), replace=True
                    )
                    bootstrap_means[b] = agg_fn(bootstrap_vals)
                return np.percentile(bootstrap_means, 97.5, axis=0)
            else:
                std_err = np.std(vals) / np.sqrt(len(vals) - 1)
                upper = agg_fn(vals) + 1.96 * std_err
                return upper

        group_df = (
            plot_df.groupby("hidden_group_channels")
            .agg({col: [aggregate_mode, lower_percentile, upper_percentile]})
            .reset_index()
        )

        # plot mean with error bars
        axs.errorbar(
            group_df["hidden_group_channels"] + eps[n],
            group_df[col][aggregate_mode],
            yerr=(
                group_df[col][aggregate_mode] - group_df[col]["lower_percentile"],
                group_df[col]["upper_percentile"] - group_df[col][aggregate_mode],
            ),
            label=plt_label,
            ls="dotted",
            marker="o",
            ms=5,
        )  # sem = standard error

    for n, group in enumerate(unique_groups):
        plot(df, n, group)
    # plot(df, n + 1)

    # standard error bars
    axs.set_title("")

    axs.set_xticks(unique_group_channels)
    axs.set_xticklabels(unique_group_channels)
    axs.set_xlabel("hidden_group_channels")
    axs.set_ylabel(col)

    axs.legend(loc="best")

    plt.show()


# using this example https://matplotlib.org/2.0.2/examples/statistics/violinplot_demo.html
def create_violin_plots(
    df: pd.DataFrame,
    col: str,
    unique_groups: List[str],
    unique_group_channels: List[int],
):
    # Create a grid of subplots
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(unique_group_channels),
        figsize=(10, 5),
        sharex=True,
        sharey=True,
    )

    df = df.copy()

    colors = ["red", "blue", "green", "brown", "purple", "orange", "yellow", "black"]
    num_needed = len(unique_group_channels) + len(unique_groups)
    colors = colors[:num_needed]

    for i, group_channels in enumerate(unique_group_channels):
        data = []
        for group in unique_groups:
            # Filter the dataframe based on the current "group" and "hidden_group_channels"
            filtered_df = df[
                (df["group"] == group) & (df["hidden_group_channels"] == group_channels)
            ]
            if len(filtered_df) == 0:
                data.append(
                    [df[df["hidden_group_channels"] == group_channels][col].mean()]
                )  # hacked fix for bad sweep
            else:
                data.append(filtered_df[col])

        (pos := [i + 1 for i in range(len(unique_groups))]).reverse()
        violinplots = []
        # note, bw_method makes "smoother" or not; settled on scott method after some trial and error.
        violinplots.append(
            axes[i].violinplot(
                data,
                pos,
                points=200,
                vert=False,
                widths=0.7,
                showmeans=True,
                showextrema=True,
                showmedians=False,
                bw_method="scott",
            )
        )
        axes[i].set_title(f"gc {group_channels}", fontsize=10)

        for vplot in violinplots:
            for patch, color in zip(vplot["bodies"], colors[: len(unique_groups)]):
                patch.set_facecolor(color)

        colors.pop(3)
        colors.insert(0, colors.pop(-1))

    ###
    # adding horizontal grid lines
    for ax in axes:
        # hide ticks on x and y axis
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # hide ticks
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")

        # set y-axis ticks to text labels
        # ax.set_yticks([1, 2, 3, 4])
        ax.set_yticks([i + 1 for i in range(len(unique_groups))])
        ax.set_yticklabels(unique_groups[::-1], fontsize=10)

        xticks = np.linspace(df[col].min(), df[col].max(), 7)
        ax.set_xticks(xticks)
        ax.xaxis.grid(True, linestyle="--", which="major", color="lightgrey")
        ax.set_xlabel(col)

    # Show the plot
    fig.suptitle("Group channels vs. " + col)
    fig.subplots_adjust(hspace=0.4)
    plt.show()


def create_swarm_plots(
    df: pd.DataFrame,
    col: str,
    unique_groups: List[str],
    unique_group_channels: List[int],
):
    # Create a grid of subplots
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(10, 5),
        sharex=True,
        sharey=True,
    )

    df = df.copy()
    sns.catplot(data=df, x=col, row="group", y="hidden_group_channels", kind="swarm")

    # Show the plot
    # fig.suptitle("Group channels vs. " + col)
    # fig.subplots_adjust(hspace=0.4)
    plt.show()


# using this example https://matplotlib.org/2.0.2/examples/statistics/violinplot_demo.html
def create_error_plots_custom(
    df: pd.DataFrame,
    target_col: str,
    covariate_col: str,
    unique_covariates: List[int],
    class_col: str,
    unique_classes: List[str],
    aggregate_mode: str = "mean",
    ax=None,
    std_error: bool = True,
    eps_noise_level: float = 0,
):
    B = 100
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(
            f"{target_col} {aggregate_mode} 95% {'distribution' if std_error else ''} interval (bootstrap)"
        )

    # for ever group in unique_groups, plot the mean of col as a function of dropout
    factor = np.max(df[covariate_col])
    eps = factor / 85 * np.linspace(-1, 1, len(unique_classes) + 1)
    eps = eps * eps_noise_level

    assert aggregate_mode in ["mean", "median", "std", "CoeffVar"]

    def mean(x):
        return np.mean(x)

    def median(x):
        return np.median(x)

    def std(x):
        return np.std(x)

    def CoeffVar(x):
        return np.std(x) / np.mean(x)

    agg_fns = {
        "mean": mean,
        "median": median,
        "std": std,
        "CoeffVar": CoeffVar,
    }
    agg_fn = agg_fns[aggregate_mode]

    def plot(df, n: int, class_setting: str = "all"):
        if class_setting == "all":
            plot_df = df
            plt_label = "all"
        else:
            plot_df = df[df[class_col] == class_setting]
            plt_label = class_setting

        def lower_percentile(vals):
            if aggregate_mode == "mean":
                # use CLT to get confidence interval
                if std_error:
                    std = np.std(vals) / np.sqrt(len(vals) - 1)
                else:
                    std = np.std(vals)
                return agg_fn(vals) - 1.96 * std
            else:
                if std_error:
                    # use bootstrap to get confidence interval
                    bootstrap_means = np.empty(B, dtype=np.float32)
                    for b in range(B):
                        bootstrap_vals = np.random.choice(
                            vals, size=len(vals), replace=True
                        )
                        bootstrap_means[b] = agg_fn(bootstrap_vals)
                    return np.percentile(bootstrap_means, 2.5, axis=0)
                else:
                    return np.percentile(vals, 2.5, axis=0)

        def upper_percentile(vals):
            if aggregate_mode == "mean":
                # use CLT to get confidence interval
                if std_error:
                    std = np.std(vals) / np.sqrt(len(vals) - 1)
                else:
                    std = np.std(vals)
                return agg_fn(vals) + 1.96 * std
            else:
                if std_error:
                    # use bootstrap to get confidence interval
                    bootstrap_means = np.empty(B, dtype=np.float32)
                    for b in range(B):
                        bootstrap_vals = np.random.choice(
                            vals, size=len(vals), replace=True
                        )
                        bootstrap_means[b] = agg_fn(bootstrap_vals)
                    return np.percentile(bootstrap_means, 97.5, axis=0)
                else:
                    return np.percentile(vals, 97.5, axis=0)

        group_df = (
            plot_df.groupby(covariate_col)
            .agg({target_col: [agg_fn, lower_percentile, upper_percentile]})
            .reset_index()
        )

        # plot mean with error bars
        ax.errorbar(
            group_df[covariate_col] + eps[n],
            group_df[target_col][aggregate_mode],
            yerr=(
                group_df[target_col][aggregate_mode]
                - group_df[target_col]["lower_percentile"],
                group_df[target_col]["upper_percentile"]
                - group_df[target_col][aggregate_mode],
            ),
            label=plt_label,
            ls="dotted",
            marker="o",
            ms=5,
        )

    for n, class_setting in enumerate(unique_classes):
        plot(df, n, class_setting)

    # standard error bars
    ax.set_title(f"{class_col}")

    # ax.set_xticks(unique_covariates)
    # ax.set_xticklabels(unique_covariates)
    ax.set_xlabel(covariate_col)
    ax.set_ylabel(target_col)

    ax.legend(loc="best")
