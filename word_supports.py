with open('appearances.txt', 'r') as f:
    supports = dict()
    for line in f.readlines():
        parts = line.split(' ')
        supports[parts[0]] = parts[1]
    print('Finish input with word END.')
    while True:
        w = input('Insert word ->')
        if w=='END':
            break
        else:
            print(w, supports.get(w,0))
