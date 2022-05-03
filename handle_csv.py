import os
import csv


if __name__ == "__main__":
    csv_path = "Z:\\private\\training\\original\\train_data.csv"
    save_path = "Z:\\private\\training\\original\\train_data2.csv"

    read = open(csv_path, 'r')
    rdr = csv.reader(read)

    lines = []
    for idx, line in enumerate(rdr):
        if idx == 0:
            lines.append(line)
            continue

        line[0] = "Z:\\private\\training\\original\\" + line[0]
        lines.append(line)

    read.close()


    write = open(save_path, 'w', newline='')
    wr = csv.writer(write)
    wr.writerows(lines)

    write.close()
