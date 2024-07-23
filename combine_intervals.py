import random


def get_random_intervals(num_intervals, max_val):
    intervals = []
    for _ in range(num_intervals):
        lower_bound = random.randint(0, max_val)
        upper_bound = random.randint(
            lower_bound, min(lower_bound + 20, max_val)
        )
        intervals.append([lower_bound, upper_bound])
    return intervals


def sort_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    return intervals


def combine_intervals(intervals):

    next_interval_idx = 1  # pointer to the next interval
    current_lower_bound = intervals[0][0]
    current_upper_bound = intervals[0][1]
    results = []

    while True:
        if next_interval_idx == len(intervals):
            results.append([current_lower_bound, current_upper_bound])
            break
        # what conditions are true as we enter this loop?
        # We have a lower and upper bound and a pointer to the next interval

        # is the next interval' s lower bound less than or equal to the current upper bound?

        if intervals[next_interval_idx][0] <= current_upper_bound:
            if intervals[next_interval_idx][1] > current_upper_bound:
                current_upper_bound = intervals[next_interval_idx][1]
            else:
                # Nothing to do here other than to move the pointer to the next interval
                # which we do at the end of the loop
                pass
        else:
            # if it's not then we are done. Append a new interval to the results using
            # the current lower and upper bounds and set the current lower and upper bounds to the next interval
            # to the next interval's lower and upper bounds
            results.append([current_lower_bound, current_upper_bound])
            current_lower_bound = intervals[next_interval_idx][0]
            current_upper_bound = intervals[next_interval_idx][1]

        next_interval_idx += 1

    return results


if __name__ == "__main__":
    num_intervals = 10
    max_val = 100
    intervals = get_random_intervals(num_intervals, max_val)
    result = combine_intervals(sort_intervals(intervals))
    print(f"Sorted intervals to combine: {sort_intervals(intervals)}")
    print(f"Combined Ranges = {result}")
