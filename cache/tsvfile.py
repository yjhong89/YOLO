import csv

anchors = {'width': [1.08,3.42,6.63,9.42,16.62],  'height':[1.19,4.41,11.38,5.11,10.52]}
field_name = ['width', 'height']

if __name__ == "__main__":
    with open('anchor_voc.tsv', 'w') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter=',', fieldnames=field_name)
        writer.writeheader()

        # Tie width and height with corresponding index
        w_h = zip(*[anchors[key] for key in field_name])

        for i, j in w_h:
            writer.writerow({field_name[0]: i, field_name[1]:j})
