class Parser:
    
    def __init__(self, path_in, path_out):
        self.path_in = path_in
        self.path_out = path_out

    @staticmethod
    def load_file():
        raise NotImplementedError


class NoahParser(Parser): 
    pass

class ChatmaniaParser(Parser):
    pass

class SwissCrawlParser(Parser):
    pass

p = Parser('', '')
p.load_file()
