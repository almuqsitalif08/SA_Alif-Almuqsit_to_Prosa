import pickle

# mambaca model dan vectorizer ke dalam varibel
filemodel = 'LinearRegression.model'
loaded_model = pickle.load(open(filemodel, 'rb'))

filevector = 'vectorizer.pickle'
loaded_vectorizer = pickle.load(open(filevector, 'rb'))

# Sentimen Analysis App
while True:
    try:
        comment = input("Masukkan komentar anda: ")
        result = loaded_model.predict(loaded_vectorizer.transform([comment]))
        if result == 'positive':
            print('Senang bisa membantu Anda. :)')
        else:
            print('Maaf atas Ketidaknyamaan ini. :(')

        lagi = input("Apakah Anda ingin memberikan komentar lagi? (y/t) \n")
        if lagi.lower() == 't':
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        print('Terima Kasih telah memberikan komentar. \nSemoga harimu menyenangkan. :)')
        break
