import csv


if __name__ == "__main__":
    with open('test.tsv', 'w') as tsvfile:
        field_name = ['width', 'height']
        writer = csv.DictWriter(tsvfile, delimiter=',', fieldnames=field_name)
        writer.writeheader()
        writer.writerow({'width':34,'height':234})
