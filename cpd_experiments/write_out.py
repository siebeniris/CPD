import json
import csv



def write_out_to_json(sentences, cpt, dates_periods, sentiment_mean, outputfile):
    """
    Write out the results to csv.
    :param sentences:
    :param cpt:
    :param dates_periods:
    :param sentiment_mean:
    :param outputfile:
    :return:
    """
    print('save to ', outputfile)
    data ={
        "sentences": sentences,
        "cpt":cpt,
        "dates_periods": dates_periods,
        "sentiment_mean": sentiment_mean
    }
    with open(outputfile, 'w') as file:
        json.dump(data, file)


def write_out_to_csv(sentences, outputfile):
    """
    Write out result sto csv.
    :param sentences:
    :param outputfile:
    :return:
    """
    print('save to ', outputfile)
    with open(outputfile, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='\"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['date', 'uid', 'lemma', 'sentence', 'sentiment', 'typical?(renovation) yes or no'])
        for idx, sentences in sentences.items():
            for el in sentences:
                csvwriter.writerow(list(el) + [''])
            csvwriter.writerow(["change point ", str(idx + 1)])

