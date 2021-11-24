import torch

# Pixel heights of each bar of Figure 23 in the slot attention paper, manually extracted from the figure pdf with a vector editing program
# I had to do this since the original numbers that produced the plot no longer existed directly, I asked Thomas Kipf and Francesco Locatello about this
# From left to right, first number in each line is the mean, second number the stdev
data = [
    205.0459, 3.9059,
    211.1575, 0.6441,
    210.9344, 0.933,
    209.5343, 1.7984,
    207.1419, 3.041,
    196.894, 6.4852,
    185.2589, 17.0437,
    179.388, 6.0363,

    188.5205, 4.9711,
    201.7532, 2.3727,
    204.5348, 1.1854,
    203.4688, 2.6886,
    202.1903, 4.2414,
    193.6525, 7.3457,
    176.3438, 18.4641,
    173.5171, 7.1016,

    121.7664, 12.7828,
    143.055, 13.8199,
    172.6887, 6.312,
    174.7183, 7.4877,
    183.4974, 6.3097,
    183.4974, 8.8239,
    103.0666, 62.4937,
    145.2499, 14.5582,

    23.4835, 6.0363,
    29.6076, 9.2967,
    54.6897, 8.3505,
    57.6678, 9.5274,
    77.3142, 10.2184,
    109.4448, 13.5,
    23.4835, 31.957,
    51.3158, 16.3336,

    1.957, 0.7102,
    2.3772, 0.9922,
    5.4227, 1.1906,
    5.7007, 1.1412,
    9.0547, 1.8591,
    17.0792, 3.5185,
    1.3046, 2.4855,
    6.0883, 2.8406,
]

data = torch.FloatTensor(data)
data = data.reshape(5, 8, 2).permute(2, 0, 1)

print(data)
data[0] *= 94.3 / data[0, 0, 0]
data[1] *= 1.1 / data[1, 0, 0]
print(data[0])
print(data[1])
