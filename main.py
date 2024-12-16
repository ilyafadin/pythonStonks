import numpy, random, os

lr = 1  # learning rate
bias = 1  # value of bias
bias2 = 1  # value of bias
biasMade = 1
biasComb = 1

# weights = [random.random(), random.random(), random.random(), random.random()] #weights generated in a list (3 weights in total for 2 neurons and the bias)
weights = [random.random(), random.random(), random.random()]
weightsMade = [random.random(), random.random(), random.random()]
weightsComb = [random.random(), random.random(), random.random()]
weights2 = [random.random(), random.random(), random.random(), random.random(),
            random.random()]  # weights generated in a list (3 weights in total for 2 neurons and the bias)
print(weights)


def sigmoid(x):
    # Our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + numpy.exp(-x))


error_tracker = []
error_tracker_Made = []
error_tracker_CombinePred = []
third_weight_tracker = []
third_weight_tracker_Made = []

def Perceptron(input1, input2, output):
    outputP = input1 * weights[0] + input2 * weights[1] + bias * weights[2]
    outputP = 1 / (1 + numpy.exp(-outputP))  # sigmoid function
    error = output - outputP
    error_tracker.append(error)
    # correction = Perceptron2(input1, input2, input3, error, 0)
    # print(correction, "correction")
    change = error * input1 * lr
    weights[0] += change
    change = error * input2 * lr
    weights[1] += change
    change = error * bias * lr
    weights[2] += change
    third_weight_tracker.append(weights[2])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import numpy as np
    import pandas as pd

    # Define converters to handle the numbers and percentage
    converters = {
        '涨跌幅': lambda x: float(x.strip('%')) / 100,
        '收盘': lambda x: float(x.replace(',', '')),
        '开盘': lambda x: float(x.replace(',', '')),
        '高': lambda x: float(x.replace(',', '')),
        '低': lambda x: float(x.replace(',', '')),
        '交易量': lambda x: x if x == "" else float(x.replace('K', '')) * 1000  # Assuming 'K' stands for thousand
    }
    # Define new column names
    new_column_names = ['Date', 'Price', 'Open', 'high', 'low', 'volume', 'change']

    # Use the `quotechar` parameter to handle quotes in the CSV
    arrr = pd.read_csv(
        'stonk/SLV.csv',
        delimiter=',',
        quotechar='"',
        thousands=',',
        converters=converters,
        names=new_column_names,
        header=0  # Replace existing header
    )

    # arrr = pd.read_csv('stonk/ethereum-20240131234902107_2.csv', delimiter=';', header="infer")


    # Reverse the DataFrame to read from bottom to top
    gold_data_reversed = arrr.iloc[::-1].reset_index(drop=True)

    print(gold_data_reversed.head(5))

    # arrr = pd.read_csv('stonk/gold.csv', delimiter=',', header="infer")  # .to_numpy()

    #date from milliseconds to date
    # from datetime import datetime
    #
    # convertDate = lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M:%S')
    # gold_data_reversed['timeOpen'] = gold_data_reversed['timeOpen'].apply(convertDate)

    open = "priceOpen"
    open = "Open"
    close = "priceClose"
    close = "Price"
    date = "timeOpen"
    date = "Date"

    range_min = 0
    range_max = 62


    def buy(row):
        if row[open] < row[close]:
            return 1
        else:
            return 0


    gold_data_reversed['stonks!'] = gold_data_reversed.apply(lambda row: buy(row), axis=1)


    # variables
    def spikes(row):
        return 1 - row[open] / row[close]


    gold_data_reversed['spikes'] = gold_data_reversed.apply(lambda row: spikes(row), axis=1)
    gold_data_reversed['cumsum'] = gold_data_reversed['spikes'].cumsum()

    # Create an array of size 200
    recent200 = np.array([])
    yesterday = ()

    # stats
    outputPs = []
    outputPSigs = []
    stonks = []

    prediction = 0

    nbd = []
    weights_tracker = [weights]
    outputPs_tracker = []
    outputPSigs_tracker = []
    outputCombPreds_tracker = []
    outputPSigsPreds_tracker = []
    prediction_tracker = []
    predicted_correctly = []


    for i, j in gold_data_reversed.head(range_max).iterrows():
        print(j[date], j[open], j[close], j['stonks!'], j['cumsum'], j['spikes'], "today is day", i)

        # calculate mean for last 200 days
        recent200 = np.append(recent200, j['spikes'])

        # calculating mean
        std = np.std(recent200)
        to_buy_next_day_std = 0
        # print("std", std)
        if j['spikes'] > recent200.mean() + std:
            # print(j['Date'], j['Open'], j['Close'])
            print("next no buy day")
            to_buy_next_day_std = -1
            nbd.append(0)
        elif j['spikes'] < recent200.mean() - std:
            # print(j['Date'], j['Open'], j['Close'])
            print("next buy day")
            # if i > 310:
            #     break
            to_buy_next_day_std = 1
            nbd.append(1)
        else:
            nbd.append(0)

        if int(i) > 0:
            Perceptron(to_buy_next_day_std, yesterday[1], gold_data_reversed.iloc[i + 1]['stonks!'])
            weights_tracker.append(list(weights))

        went_here = False
        price_change = j[close] - j[open]
        print("price_change", price_change)
        if len(recent200) > 200:
            if prediction == 1:
                went_here = True
                if prediction == j['stonks!']:
                    predicted_correctly.append(1)
                    print("predicted good for income")
                else:
                    predicted_correctly.append(-1)
                    print("predicted bad for loss")
            else:
                if j['stonks!'] == 1:
                    print("said no, was stonk!")
                    predicted_correctly.append(0)
                else:
                    print("said no, was no stonk!")
                    predicted_correctly.append(0)

            # if went_here == False:


            recent200 = recent200.take(range(len(recent200) - 200, len(recent200)))
            # prediction for tomorrow
            outputP = to_buy_next_day_std * weights[0] + j['cumsum'] * weights[1] + bias * weights[2]
            print("outputP", outputP)
            outputPs_tracker.append(outputP)
            outputPSig = sigmoid(outputP)  # ~1
            print("outputPSig", outputPSig)
            outputPSigs_tracker.append(outputPSig)
            # prediction = 1 if (outputPSig * outputPMadeSig * 10) > 0.5 else 0
            # print("prediction for tomorrow: ", prediction)
            # prediction_tracker.append(prediction)
        yesterday = (j['spikes'], j['cumsum'])
        print(yesterday, "yesterday")

    #plot the plot
    import matplotlib.pyplot as plt

    fig = plt.figure('Charts')
    ax = plt.subplot(10, 1, 1)
    ax.set_title("Open/Close")
    x_values = list(range(range_min, range_max))
    print(x_values[0])
    y_values = gold_data_reversed[open][range_min:range_max]
    print(y_values[range_min])
    plt.plot(x_values, y_values, marker='o')

    for i, txt in enumerate(x_values):
        mi = i + range_min
        # print(mi, txt)
        smth = x_values[i]
        smth2 = y_values[mi]
        ax.annotate(txt, (smth, smth2))
    y_values = gold_data_reversed[close][range_min:range_max]
    plt.plot(x_values, y_values, marker='o')
    for i, txt in enumerate(x_values):
        mi = i + range_min
        ax.annotate(txt, (x_values[i], y_values[mi]))

    # stonks!
    ax2 = fig.add_subplot(10, 1, 2)
    ax2.title.set_text('stonks!')
    y_values = gold_data_reversed["stonks!"][range_min:range_max]
    plt.plot(x_values, y_values, marker='o')

    for i, txt in enumerate(x_values):
        mi = i + range_min
        ax2.annotate(txt, (x_values[i], y_values[mi]))

    # spikes
    ax3 = fig.add_subplot(10, 1, 3)
    ax3.title.set_text('spikes')
    ax3.hlines(y=0.0, xmin=range_min, xmax=range_max, linewidth=1, color='r')
    y_values = gold_data_reversed["spikes"][range_min:range_max]
    plt.plot(x_values, y_values, marker='o')
    for i, txt in enumerate(x_values):
        mi = i + range_min
        ax3.annotate(txt, (x_values[i], y_values[mi]))

    # cumsum
    ax4 = fig.add_subplot(10, 1, 4)
    ax4.title.set_text('cumsum')
    y_values = gold_data_reversed["cumsum"][range_min:range_max]
    plt.plot(x_values, y_values, marker='o')
    for i, txt in enumerate(x_values):
        mi = i + range_min
        ax4.annotate(txt, (x_values[i], y_values[mi]))

    # outputPs
    # ax5 = fig.add_subplot(10, 1, 5)
    # ax5.title.set_text('outputPs')
    # y_values = outputPs_tracker
    # plt.plot(x_values, y_values, marker='o')
    # for i, txt in enumerate(x_values):
    #     ax5.annotate(txt, (x_values[i], y_values[i]))

    # no buy day
    ax8 = fig.add_subplot(10, 1, 5)
    ax8.title.set_text('next buy day')
    y_values = nbd[range_min:range_max]
    plt.plot(x_values, y_values, marker='o')
    for i, txt in enumerate(x_values):
        ax8.annotate(txt, (x_values[i], y_values[i]))

    # errors
    ax10 = fig.add_subplot(10, 1, 6)
    ax10.title.set_text('errors')
    y_values = error_tracker[range_min:range_max] + [0]
    plt.plot(x_values, y_values, marker='o')
    for i, txt in enumerate(x_values):
        ax10.annotate(txt, (x_values[i], y_values[i]))

    #show the plot
    plt.show()

    print("final weights", weights)
    print("final weights Made", weightsMade)
