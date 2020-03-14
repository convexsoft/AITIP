import os

class Logger:
    _files = ['err.log', 'proof.txt', 'disproof.txt', 'n.txt', 'out.log']
    _fdict = None

    def __init__(self, odir=None, arg='w+'):
        if not odir:
            return
        paths = map(lambda x: os.path.join(odir, x), self._files)
        fos = list(map(lambda x: open(x, arg), paths))
        self._fdict = dict(zip(self._files, fos))

    def log(self, string, key='out.log'):
        def complete_key(key):
            for fn in self._files:
                if key == fn or key == fn.split('.')[0]:
                    return fn

        if type(string) is not str:
            string = str(string)

        if self._fdict:
            if not key:
                key = self._default_key

            key = complete_key(key)
            fo = self._fdict.get(key)

            if not fo:
                raise ValueError('Invalid key {}'.format(key))

            fo.write(string + '\n')

        if key == 'out.log':
            print(string)

    def close(self):
        if not self._fdict:
            return
        for f in self._fdict.values():
            f.close()
