import pickle


def main():
    # load the data
    with open('file.pkl', 'rb') as f:
        myvar = pickle.load(f)


if __name__ == '__main__':
    main()
