def get_probability(x):
    return 1/5000 * x - 9/50

def get_estimate(x, y):
    sum1 = get_probability(x) * (x - 900) * (1000 -x)
    sum2 = (get_probability(x) + get_probability(y)) * (y - x) * (1000 - y)
    return sum1 + sum2

estimates = {}
lower_bound = 900
while lower_bound <= 1000:
    upper_bound = lower_bound
    while upper_bound <= 1000:
        estimates[(lower_bound, upper_bound)] = get_estimate(lower_bound, upper_bound)
        upper_bound += 0.1
    lower_bound += 0.1

ordered_estimates = sorted(estimates.items(), key=lambda x: x[1], reverse=True)
print(ordered_estimates[0])