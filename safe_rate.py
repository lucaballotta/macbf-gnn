import os
import argparse
import csv


def cal_safe_rate(args):
    log_file = os.path.join(args.path, 'test_log.csv')
    safe_traj = 0.
    all_traj = 0.
    with open(log_file) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            safe_traj += float(row[0]) * int(row[1])
            all_traj += int(row[1])
    print(f'Trajectory Number: {all_traj}, Safe Rate: {safe_traj / all_traj * 100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    cal_safe_rate(args)
