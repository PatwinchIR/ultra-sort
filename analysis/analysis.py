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
    for logline in fileinput.input():
        logline = logline.strip()
        if "elements" in logline:
            tag_fat = prevline.split(".")[1]
            if "SIMDSort64BitInteger" in tag_fat: continue
            tag = tag_fat[14:-4]
            parse_scoreline(logline)
            algo, elems, time = parse_scoreline(logline)
            num_elems = elems
            results.append([tag, elems, algo, time])
        else:
            prevline = logline

    df = DataFrame(results, columns=["Setting", "#Elements", "Algorithm", "Time"])
    print(df)

    g = sns.factorplot(x="Setting", y="Time", hue="Algorithm", data=df,
                       size=20, kind="bar", palette="muted")
    g.despine(left=True)
    g.savefig("analysis_{}.png".format(num_elems))
