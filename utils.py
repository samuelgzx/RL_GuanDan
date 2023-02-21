import logging

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)


class dotDict(dict):
    def __getattr__(self, item):
        return self.get(item)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)