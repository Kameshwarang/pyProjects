import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print('Speak Anything :')
    audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print('You said : {} '.format(text))
        if format(text)=='I am Kamesh':
            print('Your Laptop is Unlocked :: Cool ')
        else:
            print('You missed something :::: ')
    except:
        print('Sorry Could not recognize your voice')