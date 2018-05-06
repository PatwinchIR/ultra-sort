from pandas import DataFrame
import seaborn as sns
sns.set(style="whitegrid")
import fileinput

results = []

def parse_scoreline(line):
    parts = line.split()
    algo = parts[0][1:-1]
    num_elems = parts[1]
    time = parts[-2]
    return [algo, int(num_elems), float(time)]

if __name__ == '__main__':
    prevline = None
    num_elems = None
    avx_version = None
    for logline in fileinput.input():
        logline = logline.strip()
        if "elements" in logline:
            tag_fat = prevline.split(".")[1]
            avx_version = tag_fat[:6]
            tag = tag_fat[14:-4]
            parse_scoreline(logline)
            algo, elems, time = parse_scoreline(logline)
            num_elems = elems
            results.append([tag, elems, algo, time])
        else:
            prevline = logline

    df_time = DataFrame(results, columns=["Setting", "#Elements", "Algorithm", "Time(in seconds)"])
    print(df_time)

    df_only_sort = df_time[df_time['Algorithm'] == 'std::sort']
    import pandas as pd
    df_all = pd.merge(df_time, df_only_sort, on='Setting',suffixes=['', '_y'])
    df_all = df_all[[i for i in df_all.columns if i[-2:] != '_y' ] + ['Time(in seconds)_y']]
    df_all['Speedup(vs. std::sort)'] = df_all['Time(in seconds)_y']/df_all['Time(in seconds)']
    df_all = df_all.drop(["Time(in seconds)_y"], axis=1)
    print(df_all)

    FNAME = "{}_{}".format(avx_version, num_elems)
    with open("analysis_{}_time.tex".format(FNAME), "w") as f:
        f.write(df_all.to_latex())

    g = sns.factorplot(x="Setting", y="Time(in seconds)", hue="Algorithm", data=df_all,
                       size=25, kind="bar", palette="muted",)
    g.despine(left=True)
    g.fig.suptitle("{} Execution Time Comparison".format(avx_version))
    g.savefig("analysis_{}_time.png".format(FNAME))

    g = sns.factorplot(x="Setting", y="Speedup(vs. std::sort)", hue="Algorithm", data=df_all,
                       size=25, kind="bar", palette="muted")
    g.despine(left=True)
    g.fig.suptitle("{} Speedup Comparison".format(avx_version))
    g.savefig("analysis_{}_speedup.png".format(FNAME))
