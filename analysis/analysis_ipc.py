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

    df_ipc = DataFrame(results, columns=["Setting", "#Elements", "Algorithm", "IPC"])

    FNAME = "{}_{}".format(avx_version, num_elems)
    with open("analysis_{}_ipc.tex".format(FNAME), "w") as f:
        f.write(df_ipc.to_latex())

    g = sns.factorplot(x="Setting", y="IPC", hue="Algorithm", data=df_ipc,
                       size=25, kind="bar", palette="muted",)
    g.despine(left=True)
    g.fig.suptitle("{} Instructions Per Cycle Comparison".format(avx_version))
    g.savefig("analysis_{}_ipc.png".format(FNAME))