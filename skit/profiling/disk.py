#!/usr/bin/env python
"""
Adapted from: https://github.com/thodnev/MonkeyTest/tree/master
 
 -- test your hard drive read-write speed in Python
A simplistic script to show that such system programming
tasks are possible and convenient to be solved in Python

The file is being created, then written with random data, randomly read
and deleted, so the script doesn't waste your drive

(!) Be sure, that the file you point to is not something
    you need, cause it'll be overwritten during test
"""
import os, sys
from random import shuffle
import argparse
import json
from timeit import default_timer as time
import numpy as np
from tqdm import tqdm
from .timers import Ticker


def _find_file_above_threshold_size(
    directory,
    threshold_size_mb,
    recursive=True,
    timeout=10,
    limit_files=10,
    return_all=False,
    p_keep=0.1,
):
    limit_modified = int(limit_files / p_keep)

    all_files = []
    start_time = time()
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # If symbolic link, skip
            if os.path.islink(file_path):
                continue
            if os.path.getsize(file_path) > threshold_size_mb * 1024 * 1024:
                all_files.append(file_path)
        if not recursive:
            break

        if len(all_files) > limit_modified:
            break

        if time() - start_time > timeout:
            break

    if len(all_files) == 0:
        return None

    if len(all_files) > limit_files:
        shuffle(all_files)
        all_files = all_files[:limit_files]

    if return_all:
        return all_files

    fileidx = np.random.randint(0, len(all_files))
    return all_files[fileidx]


def test_file_read(
    file, block_size, blocks_count, num_reads=-1, verbose=False, timeout=30
):
    print(
        "testing file read. Block size (MB): ",
        block_size / 1024 / 1024,
        " Blocks count: ",
        blocks_count,
    )
    if num_reads == -1:
        num_reads = blocks_count

    offsets = np.arange(0, blocks_count * block_size, block_size)
    # Shuffle to avoid sequential read
    offsets = np.random.permutation(offsets)

    if num_reads > blocks_count:
        offsets = np.random.choice(offsets, num_reads, replace=True)
    else:
        offsets = offsets[:num_reads]

    f = os.open(file, os.O_RDONLY, 0o777)  # low-level I/O

    took = []
    total_t = 0

    ticker = Ticker(verbose=False)
    if verbose:
        print(
            "Reading file: {}. File size (MB): {}".format(
                file, round(os.path.getsize(file) / 1024 / 1024, 2)
            )
        )

    for offset in tqdm(offsets, desc="Reading", disable=not verbose):
        ticker.reset()

        os.lseek(f, offset, os.SEEK_SET)  # set position
        buff = os.read(f, block_size)  # read from position
        t = ticker()

        if not buff:
            raise ValueError("EOF reached. Make sure the file is large enough")
            # break  # if EOF reached

        took.append(t)
        total_t += t
        if total_t / 1000.0 > timeout:
            if verbose:
                print("Timeout reached, stopping early.")
            break

    os.close(f)
    return took


def test_random_N_file_reads(
    file,
    read_size_mb,
    block_size_kb,
    num_reads=-1,
    num_files=10,
    verbose=False,
    timeout=5,
):
    if not os.path.isdir(file):
        file = os.path.dirname(file)

    read_min_size = read_size_mb
    blocks_count = int(read_size_mb * 1024 / block_size_kb)

    files = _find_file_above_threshold_size(
        file, read_min_size, limit_files=num_files, return_all=True
    )

    if files is None:
        raise ValueError("No files found in the directory")

    took = []
    for file in tqdm(files, desc="Testing files", disable=not verbose):
        took.extend(
            test_file_read(
                file,
                int(block_size_kb * 1024),
                blocks_count,
                num_reads=num_reads,
                verbose=verbose,
                timeout=timeout / num_files,
            )
        )

    return took


def test_file_write(
    file,
    block_size_kb,
    blocks_count,
    verbose=False,
    overwrite=False,
    fsync_every=16,
):
    # Assert overwrite is False until enough testing is done to ensure safety
    assert not overwrite, "Overwrite is not well-tested yet. Please do not use it."

    # Assert the file is not already present
    if os.path.exists(file) and not overwrite:
        raise ValueError(
            "File already exists. Please provide a new file name for profiling"
        )

    f = os.open(file, os.O_CREAT | os.O_WRONLY, 0o777)  # low-level I/O

    took = []

    ticker = Ticker(verbose=False)

    block_size = int(block_size_kb * 1024)
    print("Writing blocks of size", block_size, "bytes")
    for i in tqdm(range(blocks_count), desc="Writing", disable=not verbose):
        ticker.reset()

        buff = os.urandom(block_size)
        os.write(f, buff)
        if fsync_every is None or (i % fsync_every == 0):
            os.fdatasync(f)  # force write to disk
        t = ticker()
        took.append(t)

    os.close(f)
    return took


def assert_file_writeable(file):
    if os.path.exists(file):
        raise ValueError(
            "File already exists. Please provide a new file name for profiling"
        )

    if not os.access(os.path.dirname(file), os.W_OK):
        raise ValueError("No write access to the directory")

    return True


def assert_file_readable(file):
    if not os.access(file, os.R_OK):
        raise ValueError("No read access to the file")

    return True


def assert_diskspace_available(file, size):
    if not os.path.isdir(file):
        file = os.path.dirname(file)

    stat = os.statvfs(file)
    available = stat.f_bavail * stat.f_frsize
    if available < size:
        raise ValueError("Not enough disk space available for writing the file")


def simple_profile(
    file_path,
    size=128,
    write_block_size_kb=1024,
    read_block_size_kb=1024,
    verbose=False,
    read_same_file=False,
    num_files=10,
    timeout=5,
):
    # If the path is a directory, create a file in the directory
    if os.path.isdir(file_path):
        file = os.path.join(file_path, ".testdisk.tmp")
    else:
        file = file_path
    if verbose:
        print("Writing to file: {}".format(file))

    # Make sure we have write access to the directory
    assert_file_writeable(file)
    assert_diskspace_available(file, size * 1024 * 1024)

    write_blocks = int(size * 1024 / write_block_size_kb)
    read_blocks = int(size * 1024 / read_block_size_kb)

    try:
        write_results = test_file_write(
            file, write_block_size_kb, write_blocks, verbose=verbose
        )
        if read_same_file:
            read_results = test_file_read(
                file,
                block_size=int(read_block_size_kb * 1024),
                blocks_count=read_blocks,
                verbose=verbose,
                timeout=timeout,
            )
        else:
            read_results = test_random_N_file_reads(
                file,
                size,
                block_size_kb=read_block_size_kb,
                num_files=num_files,
                verbose=verbose,
                timeout=timeout,
            )
    finally:
        if os.path.exists(file):
            os.remove(file)

    if verbose:
        print([x / 1000.0 for x in write_results], [x / 1000.0 for x in read_results])

    total_read_time = sum(read_results) / 1000.0
    total_write_time = sum(write_results) / 1000.0
    total_write_size = size
    total_read_size = len(read_results) * read_block_size_kb / 1024

    write_speed = total_write_size / total_write_time
    read_speed = total_read_size / total_read_time

    results = {
        "write": {
            "Size (MB)": size,
            "Time (sec)": round(total_write_time, 2),
            "Speed (MB/s)": round(write_speed, 2),
        },
        "read": {
            "Size (MB)": total_read_size,
            "Time (sec)": round(total_read_time, 2),
            "Speed (MB/s)": round(read_speed, 2),
        },
    }

    return results


def simple_profile_readonly(
    file, size=-1, read_block_size_kb=1024, num_reads=-1, verbose=False
):
    print("Reading from file: {}".format(file))

    # Make sure we have read access to the file
    assert_file_readable(file)

    # Get file size
    if size == -1:
        size = os.path.getsize(file) / 1024 / 1024  # in MB

    read_blocks = int(size * 1024 / read_block_size_kb)

    read_results = test_file_read(
        file,
        read_block_size_kb * 1024,
        read_blocks,
        num_reads=num_reads,
        verbose=verbose,
    )

    total_read_time = sum(read_results) / 1000.0
    total_read_size = size
    read_speed = total_read_size / total_read_time

    results = {
        "Read": {
            "Size (MB)": size,
            "Time (sec)": round(total_read_time, 2),
            "Speed (MB/s)": round(read_speed, 2),
        }
    }
    print(json.dumps(results, indent=4))

    return results


def simple_profile_N_readonly(
    file,
    read_size_mb,
    block_size_kb,
    num_reads=-1,
    num_files=10,
    verbose=False,
    timeout=5,
):

    took = test_random_N_file_reads(
        file, read_size_mb, block_size_kb, num_reads, num_files, verbose, timeout
    )
    # Metrics
    total_time = sum(took) / 1000.0
    total_size = len(took) * block_size_kb / 1024
    total_speed = total_size / total_time

    metrics = {
        "read": {
            "Size (MB)": total_size,
            "Time (sec)": round(total_time, 2),
            "Speed (MB/s)": round(total_speed, 2),
        }
    }

    return metrics


def get_args():
    parser = argparse.ArgumentParser(
        description="Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--path",
        required=False,
        action="store",
        default="./",
        help="The directory to benchmark. Please make sure the directory is on the target disk.",
    )

    parser.add_argument(
        "-s",
        "--size",
        required=False,
        action="store",
        type=int,
        default=1024,
        help="Total MB to read/write. Default is 1024 MB",
    )
    parser.add_argument(
        "-w",
        "--write-block-size",
        required=False,
        action="store",
        type=float,
        default=1.0,
        help="The block size for writing in megabytes. Default is 1 MB",
    )
    parser.add_argument(
        "-r",
        "--read-block-size",
        required=False,
        action="store",
        type=float,
        default=1.0,
        help="The block size for reading in megabytes",
    )
    parser.add_argument(
        "-j", "--json", required=False, action="store_true", help="Output to json file"
    )
    parser.add_argument(
        "-jp", "--json-path", default="results.json", help="Path to json file"
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        required=False,
        default="read",
        help="""
        Benchmark mode:
        - read: reads randomly sampled file within a directory. 
        - readwrite: writes a file within a directory, samples and reads random files within the directory.
        - readwritesame: writes a file within a directory, reads the same file again. Warning: This does not typically
        give accurate results as the file is likely to be cached in memory by the file system.
        """,
        choices=["read", "readwrite", "readwritesame"],
    )
    parser.add_argument(
        "-l",
        "--level",
        required=False,
        default=1,
        type=int,
        help="Level of benchmarking. 1: Simple, 2: Extensive",
    )
    parser.add_argument(
        "-t", "--read-timeout", type=float, default=120, help="Read timeout"
    )
    parser.add_argument(
        "-v", "--verbose", required=False, action="store_true", help="Verbose mode"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Make sure we have write access to the directory
    if not os.access(args.path, os.W_OK):
        print("Error: No write access to the directory")
        sys.exit(1)

    # Only level 1 is supported for now
    if args.level != 1:
        print("Error: Only level 1 is supported for now")
        sys.exit(1)

    if args.benchmark == "read":
        results = simple_profile_N_readonly(
            args.path,
            args.size,
            block_size=args.read_block_size * 1024,
            num_files=10,
            num_reads=-1,
            verbose=args.verbose,
            timeout=args.read_timeout,
        )
    elif args.benchmark == "readsingle":
        results = simple_profile_readonly(
            args.path,
            read_block_size_kb=args.read_block_size * 1024,
            num_reads=-1,
            verbose=args.verbose,
            timeout=args.read_timeout,
        )
    elif args.benchmark == "readwrite":
        results = simple_profile(
            args.path,
            size=args.size,
            write_block_size_kb=args.write_block_size * 1024,
            read_block_size_kb=args.read_block_size * 1024,
            verbose=args.verbose,
            timeout=args.read_timeout,
        )
    elif args.benchmark == "readwritesame":
        results = simple_profile(
            args.path,
            size=args.size,
            write_block_size_kb=args.write_block_size * 1024,
            read_block_size_kb=args.read_block_size * 1024,
            verbose=args.verbose,
            read_same_file=True,
            timeout=args.read_timeout,
        )

    # Pretty print the results
    print(json.dumps(results, indent=4))

    if args.json:
        with open(args.json_path, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
